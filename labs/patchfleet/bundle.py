"""Archive agent-generated patches using their Git base as a shared dictionary."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import tempfile

MAGIC = b"PFW1"
MODES = ("raw", "delta", "columns", "render")


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
        flags, body = bytearray(), bytearray()
        for line in split_lines(data):
            if line[:1] in (b" ", b"-", b"+"):
                flags.extend(line[:1])
                body.extend(line[1:])
            else:
                flags.extend(b"=")
                body.extend(line)
        # A final standalone prefix represents an empty unterminated body line.
        markers.append(bytes(flags))
        content.append(bytes(body))
    return pack_rows(markers + content)


def join_columns(data):
    rows = unpack_rows(data)
    if len(rows) % 2:
        raise ValueError("Malformed columns")
    n = len(rows) // 2
    result = []
    for flags, body in zip(rows[:n], rows[n:]):
        lines = split_lines(body)
        if len(lines) + 1 == len(flags):
            lines.append(b"")
        if len(lines) != len(flags):
            raise ValueError("Inconsistent column lengths")
        result.append(
            b"".join((b"" if f == 61 else bytes([f])) + s for f, s in zip(flags, lines))
        )
    return result


def rendered(basis):
    return b"".join(
        flag + basis.replace(b"\n", b"\n" + flag) for flag in (b" ", b"-", b"+")
    )


def transform(mode, rows):
    return separate(rows) if mode == "columns" else pack_rows(rows)


def zstd(data, base=None, decode=False):
    with tempfile.TemporaryDirectory() as tmp:
        command = ["zstd", "-q", "-c"]
        if base:
            path = Path(tmp) / "base"
            path.write_bytes(base)
            command.append(f"--patch-from={path}")
        command += ["-d"] if decode else ["-19", f"--stream-size={len(data)}"]
        return subprocess.run(
            command, input=data, check=True, capture_output=True
        ).stdout


def encode(rows, manifest, basis, *, strategy="auto"):
    raw = pack_rows(rows)
    digest = hashlib.sha256(raw).digest()
    candidates = {}
    if strategy != "auto" and strategy not in MODES:
        raise ValueError("Unknown encoder strategy")
    for mode in MODES if strategy == "auto" else (strategy,):
        b = None if mode == "raw" else rendered(basis) if mode == "render" else basis
        encoded = zstd(transform(mode, rows), b)
        meta = (
            b""
            if mode == "raw"
            else json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        )
        header = zstd(meta) if meta else b""
        candidates[mode] = (
            MAGIC
            + bytes([MODES.index(mode)])
            + digest
            + struct.pack(">I", len(header))
            + header
            + encoded
        )
    chosen = min(candidates, key=lambda key: len(candidates[key]))
    return candidates[chosen], {key: len(val) for key, val in candidates.items()}


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
        for line in split_lines(row):
            if line.startswith(b"--- a/"):
                try:
                    path = line[6:].rstrip(b"\r\n").split(b"\t", 1)[0].decode()
                except UnicodeDecodeError:
                    continue
                rel = PurePosixPath(path)
                if path and not rel.is_absolute() and ".." not in rel.parts:
                    paths.add(path)
    return sorted(paths)


def git_source(directory, project, commit, path):
    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", project
    ) or ".." in project.split("/"):
        raise ValueError("Expected owner/repository")
    if not re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit):
        raise ValueError("Expected full Git commit ID")
    repo = directory / project
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    r = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", commit + ":" + path],
        capture_output=True,
    )
    return r.stdout if r.returncode == 0 else None


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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=("pack", "unpack"))
    p.add_argument("repos", type=Path, help="Git clones named owner/repository")
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--mode",
        choices=("auto", *MODES),
        default="auto",
        help="Select a representation directly or compare all four",
    )
    args = p.parse_args()
    reader = lambda owner, base, path: git_source(args.repos, owner, base, path)
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
        meta, rows, basis = collect(groups, reader)
        archive, sizes = encode(rows, meta, basis, strategy=args.mode)
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
