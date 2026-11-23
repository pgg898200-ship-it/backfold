"""Archive agent-generated patches using their Git base as a shared dictionary."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tempfile

MAGIC = b"PFW1"
MODES = ("raw", "delta", "columns", "render")
PREFIXES = tuple(b"" if code == ord("=") else bytes([code]) for code in range(256))


def pack_rows(rows):
    return b"".join(struct.pack(">I", len(x)) + x for x in rows)


def unpack_rows(data):
    rows = []
    offset = 0
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("Incomplete record length")
        size = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        if size > len(data) - offset:
            raise ValueError("Incomplete record")
        rows.append(data[offset : offset + size])
        offset += size
    return rows


def split_lines(data):
    pieces = data.split(b"\n")
    return [part + b"\n" for part in pieces[:-1]] + ([pieces[-1]] if pieces[-1] else [])


def separate(rows):
    markers, content = [], []
    for data in rows:
        lines = data.split(b"\n")
        flags = bytearray()
        # split leaves a sentinel after a terminating newline (and for an empty patch).
        count = len(lines) if lines[-1] else len(lines) - 1
        for i in range(count):
            line = lines[i]
            if line and line[0] in b" +-":
                flags.append(line[0])
                lines[i] = line[1:]
            else:
                flags.append(ord("="))
        markers.append(flags)
        content.append(b"\n".join(lines))
    return pack_rows(markers + content)


def join_columns(data):
    rows = unpack_rows(data)
    if len(rows) % 2:
        raise ValueError("Malformed columns")
    n = len(rows) // 2
    result = []
    for flags, body in zip(rows[:n], rows[n:]):
        lines = body.split(b"\n")
        # A lone final marker has an empty body and no terminating newline.
        if len(lines) == len(flags):
            tail = b""
        elif len(lines) == len(flags) + 1 and not lines[-1]:
            lines.pop()
            tail = b"\n" if flags else b""
        else:
            raise ValueError("Inconsistent column lengths")
        result.append(
            b"\n".join(PREFIXES[f] + line for f, line in zip(flags, lines)) + tail
        )
    return result


def rendered(basis):
    return b"".join(
        flag + basis.replace(b"\n", b"\n" + flag) for flag in (b" ", b"-", b"+")
    )


def zstd(data, base=None, decode=False, *, level=19):
    with tempfile.TemporaryDirectory() as tmp:
        command = ["zstd", "-q", "-c"]
        if base:
            path = Path(tmp) / "base"
            path.write_bytes(base)
            command.append(f"--patch-from={path}")
        if decode:
            command.append("-d")
        else:
            if level > 19:
                command.extend(["--ultra", "--long=27"])
            command.extend([f"-{level}", f"--stream-size={len(data)}"])
        return subprocess.run(
            command, input=data, check=True, capture_output=True
        ).stdout


def encode(rows, manifest, basis, *, strategy="auto", level=19):
    if strategy != "auto" and strategy not in MODES:
        raise ValueError("Unknown encoder strategy")
    raw = pack_rows(rows)
    digest = hashlib.sha256(raw).digest()
    header = b""
    if strategy != "raw":
        meta = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        header = zstd(meta)
    sizes = {}
    best = None
    for mode in MODES if strategy == "auto" else (strategy,):
        reference = None if mode == "raw" else basis
        if mode == "render":
            reference = rendered(basis)
        data = separate(rows) if mode == "columns" else raw
        payload = zstd(data, reference, level=level)
        lookup = b"" if mode == "raw" else header
        size = 41 + len(lookup) + len(payload)
        sizes[mode] = size
        if best is None or size < len(best):
            best = b"".join(
                (
                    MAGIC,
                    bytes([MODES.index(mode)]),
                    digest,
                    struct.pack(">I", len(lookup)),
                    lookup,
                    payload,
                )
            )
    return best, sizes


def decode(bundle, source):
    if len(bundle) < 41 or bundle[:4] != MAGIC or bundle[4] >= len(MODES):
        raise ValueError("Invalid patch archive")
    mode = MODES[bundle[4]]
    size = struct.unpack_from(">I", bundle, 37)[0]
    m = bundle[41 : 41 + size]
    basis = source(json.loads(zstd(m, decode=True))) if m else b""
    if mode == "render":
        basis = rendered(basis)
    recovered = zstd(bundle[41 + size :], basis or None, True)
    rows = join_columns(recovered) if mode == "columns" else unpack_rows(recovered)
    if hashlib.sha256(pack_rows(rows)).digest() != bundle[5:37]:
        raise ValueError("Reconstructed patches failed SHA-256 verification")
    return rows


def names(rows):
    paths = set()
    for row in rows:
        for match in re.finditer(rb"^--- a/(.*)$", row, re.MULTILINE):
            try:
                path = match[1].rstrip(b"\r").split(b"\t", 1)[0].decode()
            except UnicodeDecodeError:
                continue
            rel = PurePosixPath(path)
            if path and not rel.is_absolute() and ".." not in rel.parts:
                paths.add(path)
    return sorted(paths)


class GitSource:
    """Read pinned blobs through one Git process for the current repository."""

    def __init__(self, directory):
        self.directory = directory
        self.project = None
        self.process = None

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.wait()
            self.process = self.project = None

    def __call__(self, project, commit, path):
        if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit):
            raise ValueError("Expected full Git commit ID")
        if self.project != project:
            if not re.fullmatch(
                r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", project
            ) or ".." in project.split("/"):
                raise ValueError("Expected owner/repository")
            repo = self.directory / project
            if not repo.is_dir():
                raise FileNotFoundError(repo)
            self.close()
            proc = subprocess.Popen(
                ["git", "-C", str(repo), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            self.process = proc
            self.project = project
        query = (commit + ":" + path).encode()
        if b"\0" in query:
            raise ValueError("NUL in Git lookup path")
        if b"\n" in query:
            # A legacy manifest can refer to a filename outside the batch line protocol.
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.directory / project),
                    "cat-file",
                    "blob",
                    commit + ":" + path,
                ],
                capture_output=True,
            )
            return result.stdout if result.returncode == 0 else None
        proc = self.process
        proc.stdin.write(query + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline()
        if header == query + b" missing\n":
            return None
        fields = header.split()
        if len(fields) != 3:
            raise ValueError("Unexpected Git batch response")
        size = int(fields[2])
        data = proc.stdout.read(size)
        if len(data) != size or proc.stdout.read(1) != b"\n":
            raise ValueError("Incomplete Git object")
        return data if fields[1] == b"blob" else None


def git_source(directory, project, commit, path):
    with closing(GitSource(directory)) as source:
        return source(project, commit, path)


def collect(groups, source):
    manifest, rows, basis = [], [], bytearray()
    for project, commit, patches in groups:
        resolved = []
        for path in names(patches):
            data = source(project, commit, path)
            if data:
                resolved.append(path)
                basis.extend(data)
        manifest.append([project, commit, resolved])
        rows.extend(patches)
    return manifest, rows, bytes(basis)


def dictionary(manifest, source):
    out = bytearray()
    for project, commit, paths in manifest:
        for path in paths:
            data = source(project, commit, path)
            if data is None:
                raise ValueError(f"Missing base {project} {commit} {path}")
            out.extend(data)
    return bytes(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version", action="version", version="Patchfleet Archive 1.0.0 (PFW1)"
    )
    p.add_argument("operation", choices=("pack", "unpack"))
    p.add_argument("repos", type=Path, help="Git clones named owner/repository")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--mode",
        choices=("auto", *MODES),
        help="Default: columns with a Git base, raw otherwise. Use auto to compare all four.",
    )
    p.add_argument(
        "--level",
        type=int,
        choices=range(1, 23),
        metavar="1..22",
        default=19,
        help="zstd payload compression level (default 19, compact 22, fast 12)",
    )
    args = p.parse_args()
    with closing(GitSource(args.repos)) as reader:
        run(args, reader)


def run(args, reader):
    if args.operation == "pack":
        spec = json.loads(args.input.read_text())
        groups = [
            (
                r["repo"],
                r["commit"],
                [(args.input.parent / f).read_bytes() for f in r["patches"]],
            )
            for r in spec
        ]
        if args.mode == "raw":
            meta, rows, basis = (
                [],
                [row for _, _, patches in groups for row in patches],
                b"",
            )
        else:
            meta, rows, basis = collect(groups, reader)
        mode = args.mode or ("columns" if basis else "raw")
        archive, sizes = encode(rows, meta, basis, strategy=mode, level=args.level)
        if decode(archive, lambda m: dictionary(m, reader)) != rows:
            raise ValueError("Round trip failed")
        with args.output.open("xb") as f:
            f.write(archive)
        print(json.dumps(sizes))
    else:
        rows = decode(args.input.read_bytes(), lambda m: dictionary(m, reader))
        args.output.mkdir(exist_ok=False)
        for i, data in enumerate(rows):
            (args.output / f"{i:06}.patch").write_bytes(data)


if __name__ == "__main__":
    main()
