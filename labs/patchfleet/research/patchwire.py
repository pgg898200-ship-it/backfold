"""Losslessly encode unified patches using original source lines as side information."""

import re
from pathlib import PurePosixPath

HUNK = re.compile(rb"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")


def source_path(line):
    if not line.startswith(b"--- a/"):
        return None
    name = line[6:].rstrip(b"\r\n").split(b"\t")[0].decode("utf8", "surrogateescape")
    rel = PurePosixPath(name)
    return name if not rel.is_absolute() and ".." not in rel.parts else None


def lines(data):
    chunks = data.split(b"\n")
    return [x + b"\n" for x in chunks[:-1]] + ([chunks[-1]] if chunks[-1] else [])


class Cursor:
    def __init__(self, sources):
        self.sources = sources
        self.source = []
        self.index = -1

    def literal(self, line):
        # A deleted source line may itself start with '-- a/'. Only recognize
        # file headers outside hunks or after the previous hunk consumed its base.
        if line.startswith(b"diff --git "):
            self.index, self.source = -1, []
            return
        if line.startswith(b"@@"):
            h = HUNK.match(line)
            self.index = int(h[1]) - 1 if h else -1
        elif self.index < 0 and line.startswith(b"--- "):
            self.source = self.sources.get(source_path(line), [])
        elif self.index >= 0 and line[:1] in (b"-", b" "):
            self.index += 1


def length(n):
    data = bytearray()
    while n >= 128:
        data.append((n & 127) | 128)
        n >>= 7
    data.append(n)
    return bytes(data)


def encode(patch, sources):
    cursor = Cursor(sources)
    out = bytearray()
    run, marker = 0, b""
    for line in [*lines(patch), b""]:
        flag = line[:1]
        hit = (
            flag in (b" ", b"-")
            and 0 <= cursor.index < len(cursor.source)
            and line[1:] == cursor.source[cursor.index]
        )
        if run and (not hit or marker != flag):
            out.extend(b"\0" + marker + length(run))
            run = 0
        if hit:
            run, marker = run + 1, flag
            cursor.index += 1
        else:
            out.extend(line.replace(b"\0", b"\0L"))
            cursor.literal(line)
    return bytes(out)


def decode(data, sources):
    out = bytearray()
    cursor = Cursor(sources)
    i = 0
    while i < len(data):
        if data[i : i + 2] in (b"\0 ", b"\0-"):
            marker = data[i + 1 : i + 2]
            i += 2
            count, shift = 0, 0
            while True:
                byte = data[i]
                i += 1
                count |= (byte & 127) << shift
                if byte < 128:
                    break
                shift += 7
            if (
                count <= 0
                or cursor.index < 0
                or cursor.index + count > len(cursor.source)
            ):
                raise ValueError("Invalid source reference")
            for line in cursor.source[cursor.index : cursor.index + count]:
                out.extend(marker + line)
            cursor.index += count
        else:
            end = data.find(b"\n", i)
            end = len(data) if end < 0 else end + 1
            line = data[i:end].replace(b"\0L", b"\0")
            out.extend(line)
            cursor.literal(line)
            i = end
    return bytes(out)
