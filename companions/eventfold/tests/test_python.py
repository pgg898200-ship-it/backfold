import io
from pathlib import Path
import random
import subprocess
import sys
import unittest

from eventfold import (
    FrameTooLarge, FormatError, MAGIC, decode_frames, encode_frames, pack,
    pack_stream, split_sse, unpack, unpack_stream,
)


class CodecTests(unittest.TestCase):
    def test_record_vectors(self):
        self.assertEqual(MAGIC, pack(b""))
        self.assertEqual(list(decode_frames([MAGIC + b"\x00\x80"])), [b"", b""])
        self.assertEqual(list(encode_frames([b"", b"X", b"X"])),
                         [MAGIC, b"\x80", b"\x01X", b"\x80"])
        seq = [b"startABCend", b"startZend"]
        self.assertEqual(list(encode_frames(seq))[-1], b"\x41\x05\x03Z")
        # Literal and patch tie at three bytes. The literal wins.
        self.assertEqual(list(encode_frames([b"xyz", b"xy"]))[-1], b"\x02xy")

    def test_middle_patch(self):
        stationary = b"--shared-envelope-field--"
        old = b"PREFIX:" + b"old" + stationary + b"abc" + b":SUFFIX"
        new = b"PREFIX:" + b"new!" + stationary + b"xyz" + b":SUFFIX"
        records = list(encode_frames([old, new]))
        self.assertEqual(records[-1], bytes([129, 7, 7, 3, len(stationary), 4, 3]) + b"new!xyz")
        self.assertEqual(list(decode_frames(records)), [old, new])
        for payload in [
                bytes([129, 0, 0, 0, 1, 0, 0]),
                bytes([129, 0, 0, 1, 0, 0, 0]),
                bytes([129, 0, 0, 0, 0, 1, 1]) + b"x",
                bytes([129, 0, 0, 0, 0, 128, 0, 0]),
        ]:
            with self.assertRaises(FormatError):
                unpack(MAGIC + payload)
        with self.assertRaises(FrameTooLarge):
            unpack(MAGIC + bytes([129, 0, 0, 0, 0, 2, 2]), max_frame_size=3)

    def test_all_chunk_boundaries(self):
        data = b":hi\r\n\r\ndata: \xff\n\n\rX\r\rdata:\x00\r\n\npartial\xff"
        expected = [b":hi\r\n\r\n", b"data: \xff\n\n", b"\r", b"X\r\r", b"data:\x00\r\n\n", b"partial\xff"]
        for i in range(len(data) + 1):
            self.assertEqual(list(split_sse([data[:i], b"", data[i:]])), expected)
        self.assertEqual(list(split_sse(bytes([c]) for c in data)), expected)
        packed = pack(data)
        self.assertEqual(list(decode_frames(bytes([c]) for c in packed)), expected)
        self.assertEqual(unpack(packed), data)

    def test_partial_lines_and_delimiters(self):
        for data, result in [
            (b"", []), (b"\r", [b"\r"]), (b"\r\r", [b"\r", b"\r"]),
            (b"a\r\n", [b"a\r\n"]), (b"\r\n\n", [b"\r\n", b"\n"]),
            (b"a\n\r", [b"a\n\r"]), (b"a\nb\rc", [b"a\nb\rc"]),
        ]:
            self.assertEqual(list(split_sse(bytes([c]) for c in data)), result)

    def test_extended_lengths(self):
        for n in [62, 63, 64, 127, 128, 190, 191, 16384]:
            frames = [b"x"*n, b"x"*n+b"y", b"z"+b"x"*n+b"y"]
            wire = b"".join(encode_frames(frames))
            self.assertEqual(list(decode_frames([wire])), frames)

    def test_invalid_streams(self):
        for wire in [b"", b"EFL", b"EVF1"]:
            with self.assertRaises(FormatError):
                unpack(wire)
        for record in [
            b"\xff", b"\x04abc", b"\x40", b"\x40\x00", b"\x40\x01\x00",
            b"\x01a\x40\x01\x01", b"\x3f\x80\x00", b"\x3f"+b"\xff"*9+b"\x02",
            b"\x3f"+b"\xff"*9+b"\x01", b"\x40\x80\x00\x00",
        ]:
            with self.subTest(record=record), self.assertRaises(FormatError):
                unpack(MAGIC + record)

    def test_limit(self):
        self.assertEqual(unpack(pack(b"a\n\n", max_frame_size=3)), b"a\n\n")
        for data in [b"\r\n", b"ab", b"a\r", b"\nabc"]:
            with self.assertRaises(FrameTooLarge):
                pack(data, max_frame_size=1)
        with self.assertRaises(FrameTooLarge):
            list(encode_frames([b"ab"], max_frame_size=1))
        for record in [b"\x02", b"\x01a\x41\x01\x00", b"\x3f\x01"]:
            with self.assertRaises(FrameTooLarge):
                unpack(MAGIC + record, max_frame_size=1)
        for bad in [-1, 1 << 64]:
            with self.assertRaises(ValueError):
                pack(b"", max_frame_size=bad)

    def test_first_frame_is_delivered_before_next_read(self):
        def guarded(first):
            yield first
            raise AssertionError("read beyond the first complete event")

        self.assertEqual(next(split_sse(guarded(b"data: x\n\n"))), b"data: x\n\n")
        self.assertEqual(next(decode_frames(guarded(MAGIC + b"\x01z"))), b"z")

    def test_binary_mutations(self):
        rng = random.Random(42)
        frames = [b""]
        for _ in range(500):
            old = frames[-1]
            a, b = sorted(rng.choices(range(len(old) + 1), k=2))
            frames.append(old[:a] + rng.randbytes(rng.randrange(72)) + old[b:])
        wire = b"".join(encode_frames(frames))
        self.assertEqual(list(decode_frames(wire[i:i+7] for i in range(0,len(wire),7))), frames)

    def test_flush_and_short_writes(self):
        class Sink(io.BytesIO):
            def __init__(self):
                super().__init__()
                self.flushes = 0

            def write(self, data):
                return super().write(data[:2])

            def flush(self):
                self.flushes += 1
                return super().flush()

        first = Sink()
        raw = b"x\n\ny\n\n:tail"
        pack_stream(io.BytesIO(raw), first)
        self.assertEqual(first.getvalue(), pack(raw))
        self.assertEqual(first.flushes, 4)
        second = Sink()
        unpack_stream(io.BytesIO(first.getvalue()), second)
        self.assertEqual(second.getvalue(), raw)
        self.assertEqual(second.flushes, 3)

    def test_fixtures(self):
        for path in sorted((Path(__file__).resolve().parents[1] / "fixtures").glob("*.sse")):
            with self.subTest(path=path.name):
                data = path.read_bytes()
                self.assertEqual(unpack(pack(data)), data)

    def test_python_cli(self):
        data = b"a\r\n\r\n:tail\x00\xff"
        cmd = [sys.executable, "-m", "eventfold"]
        encoded = subprocess.run(cmd+["pack"], input=data, capture_output=True, check=True)
        self.assertEqual(encoded.stdout, pack(data))
        decoded = subprocess.run(cmd+["unpack"], input=encoded.stdout, capture_output=True, check=True)
        self.assertEqual(decoded.stdout, data)
        invalid = subprocess.run(cmd+["unpack"], input=b"!", capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertFalse(invalid.stdout)
