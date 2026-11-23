"""Charge stock codecs for output only, even when they require the same source basis."""

import json
import lzma
from pathlib import Path
import subprocess
import sys
import brotli
import bsdiff4


def compare(work, target, xd):
    basis = work / "heldout.base"
    source = basis.read_bytes()
    path = work / "heldout.raw"
    raw = path.read_bytes()
    result = {}
    for name, flags in (
        ("zstd22", []),
        ("zstd22_source", [f"--patch-from={basis}"]),
        ("zstd22_long_source", ["--long=27", f"--patch-from={basis}"]),
    ):
        payload = subprocess.check_output(
            ["zstd", "-q", "--ultra", "-22", *flags, "-c", str(path)]
        )
        assert (
            subprocess.check_output(["zstd", "-q", "-d", *flags, "-c"], input=payload)
            == raw
        )
        result[name] = len(payload)
    delta = subprocess.check_output(
        [str(xd), "-9", "-S", "none", "-e", "-s", str(basis), "-c", str(path)]
    )
    assert (
        subprocess.check_output([str(xd), "-d", "-s", str(basis), "-c"], input=delta)
        == raw
    )
    result["xdelta3_322_then_xz9"] = len(lzma.compress(delta, preset=9))
    payload = bsdiff4.diff(source, raw)
    assert bsdiff4.patch(source, payload) == raw
    result["bsdiff4_126"] = len(payload)
    payload = brotli.compress(raw, quality=11)
    assert brotli.decompress(payload) == raw
    result["brotli11"] = len(payload)
    target.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    compare(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
