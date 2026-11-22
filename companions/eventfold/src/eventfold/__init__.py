"""Streaming, byte-exact EFL1 transport for raw server-sent events."""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import BinaryIO

MAGIC = b"EFL1"
DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024
_CHUNK = 64 * 1024
_LINE_END = re.compile(rb"\r\n|[\r\n]")
_UINT64_MAX = (1 << 64) - 1


class FormatError(ValueError):
    """The input is not a complete, well-formed EFL1 stream."""


class FrameTooLarge(ValueError):
    """A raw frame exceeds the configured limit."""


def _check_limit(limit: int) -> None:
    if not isinstance(limit, int) or limit < 0 or limit > _UINT64_MAX:
        raise ValueError("max_frame_size must be an unsigned 64-bit integer")


def _view(data: bytes) -> memoryview:
    return memoryview(data).cast("B")


class _SSESplitter:
    """Incremental raw frame splitter shared by blocking and ASGI transports."""

    def __init__(self, max_frame_size: int):
        _check_limit(max_frame_size)
        self.limit = max_frame_size
        self.pending = bytearray()
        self.line_start = 0
        self.scan = 0

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        pending = self.pending
        view = _view(chunk)
        for offset in range(0, len(view), _CHUNK):
            pending.extend(view[offset:offset + _CHUNK])
            block_start = 0
            for match in _LINE_END.finditer(pending, self.scan):
                end = match.end()
                # Resolve a trailing CR when the next byte or EOF arrives.
                if end == len(pending) and pending[end - 1] == 13:
                    self.scan = match.start()
                    break
                if end - block_start > self.limit:
                    raise FrameTooLarge("raw SSE frame exceeds max_frame_size")
                if match.start() == self.line_start:
                    yield bytes(pending[block_start:end])
                    block_start = end
                self.line_start = end
            else:
                self.scan = len(pending)
            if len(pending) - block_start > self.limit:
                raise FrameTooLarge("raw SSE frame exceeds max_frame_size")
            if block_start:
                del pending[:block_start]
                self.line_start -= block_start
                self.scan -= block_start

    def finish(self) -> Iterator[bytes]:
        if self.pending:
            yield bytes(self.pending)


def split_sse(chunks: Iterable[bytes], *,
              max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> Iterator[bytes]:
    """Yield raw blocks through each SSE blank line, including its terminator.

    Accept arbitrary input chunk boundaries and LF, CR, or CRLF line endings.
    Keep comments, blank blocks, arbitrary byte values and an unfinished EOF tail.
    No event data is decoded or reserialized.
    """
    splitter = _SSESplitter(max_frame_size)
    for chunk in chunks:
        yield from splitter.feed(chunk)
    yield from splitter.finish()


def _uleb(n: int) -> bytes:
    result = bytearray()
    while n >= 128:
        result.append((n & 127) | 128)
        n >>= 7
    result.append(n)
    return bytes(result)


def _tag(kind: int, size: int) -> bytes:
    return bytes([kind | min(size, 63)]) + (_uleb(size - 63) if size >= 63 else b"")


def _edges(previous: bytes, frame: bytes) -> tuple[int, int]:
    limit = min(len(previous), len(frame))
    p = 0
    s = 0
    a, b = _view(previous), _view(frame)
    while p + 64 <= limit and a[p:p + 64] == b[p:p + 64]:
        p += 64
    while p < limit and a[p] == b[p]:
        p += 1
    # Bound both suffixes by the unused bytes after their shared prefix.
    ae, be = len(a), len(b)
    while s + 64 <= limit - p and a[ae-s-64:ae-s] == b[be-s-64:be-s]:
        s += 64
    while s < limit - p and a[ae-s-1] == b[be-s-1]:
        s += 1
    return p, s


def _middle(previous: bytes, frame: bytes, p: int, s: int) -> tuple[int, int, int] | None:
    old_size = len(previous) - p - s
    size = len(frame) - p - s
    if min(old_size, size) < 24 or p + s < 8:
        return None
    span = size - 8
    for anchor in (span // 2, span // 4, span - span // 4):
        low = p + max(0, anchor - 32)
        high = p + min(old_size - 8, anchor + 32) + 8
        match = previous.find(frame[p+anchor:p+anchor+8], low, high)
        if match < 0:
            continue
        old = match - p
        left = anchor
        n = 8
        while old and left and previous[p+old-1] == frame[p+left-1]:
            old -= 1
            left -= 1
            n += 1
        while old+n < old_size and left+n < size and previous[p+old+n] == frame[p+left+n]:
            n += 1
        if n >= 16:
            return old, left, n
    return None


def _encode_python(previous: bytes, frame: bytes) -> bytes:
    if previous == frame:
        return b"\x80"
    literal = _tag(0, len(frame))
    p, s = _edges(previous, frame)
    n = len(frame) - p - s
    patch = _tag(64, n) + _uleb(p) + _uleb(s)
    patch_cost = len(patch) + n
    literal_cost = len(literal) + len(frame)
    middle = _middle(previous, frame, p, s)
    if middle is not None:
        skip, left, middle_len = middle
        right = n - left - middle_len
        fields = (p, s, skip, middle_len, left, right)
        header = b"\x81" + b"".join(_uleb(value) for value in fields)
        if len(header) + left + right < min(patch_cost, literal_cost):
            return header + frame[p:p+left] + frame[p+left+middle_len:len(frame)-s]
    if patch_cost < literal_cost:
        return patch + frame[p:len(frame) - s]
    return literal + frame


try:
    from ._native import encode as _encode
except ImportError:
    _encode = _encode_python


def encode_frames(frames: Iterable[bytes], *,
                  max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> Iterator[bytes]:
    """Yield the EFL1 header and one complete record per supplied raw frame.

    Unlike split_sse(), this API accepts already framed messages including empty
    ones. The iterable cannot be a single byte string.
    """
    _check_limit(max_frame_size)
    yield MAGIC
    previous = b""
    for frame in frames:
        view = _view(frame)
        if len(view) > max_frame_size:
            raise FrameTooLarge("raw frame exceeds max_frame_size")
        frame = frame if type(frame) is bytes else bytes(view)
        yield _encode(previous, frame)
        previous = frame


class _Reader:
    """Consume chunked input, retaining at most the caller's active chunk."""

    def __init__(self, chunks: Iterable[bytes]):
        self.chunks = iter(chunks)
        self.chunk = memoryview(b"")

    def advance(self) -> bool:
        while not self.chunk:
            chunk = next(self.chunks, None)
            if chunk is None:
                return False
            self.chunk = _view(chunk)
        return True

    def take(self, size: int) -> bytes:
        if size <= len(self.chunk):
            result = bytes(self.chunk[:size])
            self.chunk = self.chunk[size:]
            return result
        result = bytearray()
        while len(result) < size:
            if not self.advance():
                raise FormatError("truncated EFL1 record or header")
            count = min(size - len(result), len(self.chunk))
            result.extend(self.chunk[:count])
            self.chunk = self.chunk[count:]
        return bytes(result)

    def byte(self) -> int:
        if not self.advance():
            raise FormatError("truncated EFL1 record")
        result = self.chunk[0]
        self.chunk = self.chunk[1:]
        return result

    def uleb(self) -> int:
        value = 0
        for shift in range(0, 64, 7):
            b = self.byte()
            if shift == 63 and b > 1:
                raise FormatError("unsigned LEB128 overflow")
            value |= (b & 127) << shift
            if b < 128:
                if shift and b == 0:
                    raise FormatError("noncanonical unsigned LEB128")
                return value
        raise FormatError("unsigned LEB128 overflow")


def decode_frames(chunks: Iterable[bytes], *,
                  max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> Iterator[bytes]:
    """Decode one EFL1 stream and yield each frame as soon as it arrives.

    A clean EOF at a record boundary is valid. Authentication, integrity and
    detection of a missing whole terminal record belong to the transport.
    """
    _check_limit(max_frame_size)
    reader = _Reader(chunks)
    if reader.take(len(MAGIC)) != MAGIC:
        raise FormatError("expected EFL1 magic")
    previous = b""
    while reader.advance():
        tag = reader.byte()
        if tag > 129:
            raise FormatError("reserved EFL1 record tag")
        if tag == 129:
            p, s, skip, middle, left, right = (reader.uleb() for _ in range(6))
            if p + skip + middle + s > len(previous):
                raise FormatError("middle patch exceeds previous frame bounds")
            if p + s + middle + left + right > max_frame_size:
                raise FrameTooLarge("decoded frame exceeds max_frame_size")
            ref = memoryview(previous)
            previous = b"".join((ref[:p], reader.take(left),
                                 ref[p+skip:p+skip+middle], reader.take(right),
                                 ref[len(ref)-s:]))
            del ref
        elif tag != 128:
            size = tag & 63
            if size == 63:
                size += reader.uleb()
                if size > _UINT64_MAX:
                    raise FormatError("EFL1 record length overflow")
            if size > max_frame_size:
                raise FrameTooLarge("EFL1 record exceeds max_frame_size")
            if tag & 64:
                p, s = reader.uleb(), reader.uleb()
                if p + s > len(previous):
                    raise FormatError("patch exceeds previous frame bounds")
                if p + s + size > max_frame_size:
                    raise FrameTooLarge("decoded frame exceeds max_frame_size")
                previous = b"".join((memoryview(previous)[:p], reader.take(size),
                                     memoryview(previous)[len(previous)-s:]))
            else:
                previous = reader.take(size)
        yield previous


def pack(data: bytes, *, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> bytes:
    """Pack a complete raw SSE byte string. For live inputs use pack_stream."""
    frames = split_sse((data,), max_frame_size=max_frame_size)
    return b"".join(encode_frames(frames, max_frame_size=max_frame_size))


def unpack(data: bytes, *, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> bytes:
    """Unpack a complete EFL1 byte string into its original SSE bytes."""
    return b"".join(decode_frames((data,), max_frame_size=max_frame_size))


def _chunks(source: BinaryIO) -> Iterator[bytes]:
    # BufferedReader.read(n) may wait for n bytes even after a complete event.
    # HTTPResponse.read1 performs at most one underlying buffered read.
    read = getattr(source, "read1", source.read)
    while chunk := read(_CHUNK):
        yield chunk


def _write(chunks: Iterable[bytes], target: BinaryIO, flush: bool) -> None:
    for chunk in chunks:
        rest = memoryview(chunk)
        while rest:
            count = target.write(rest)
            if count is None or count <= 0:
                raise OSError("blocking output stream could not make progress")
            rest = rest[count:]
        if flush:
            target.flush()


def pack_stream(source: BinaryIO, target: BinaryIO, *,
                max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
                flush: bool = True) -> None:
    """Pack blocking binary streams. By default flush every emitted record."""
    frames = split_sse(_chunks(source), max_frame_size=max_frame_size)
    _write(encode_frames(frames, max_frame_size=max_frame_size), target, flush)


def unpack_stream(source: BinaryIO, target: BinaryIO, *,
                  max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
                  flush: bool = True) -> None:
    """Decode blocking binary streams and flush every frame by default."""
    _write(decode_frames(_chunks(source), max_frame_size=max_frame_size), target, flush)
