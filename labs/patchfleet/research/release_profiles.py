"""Measure PFW1 release profiles against the pinned original records and reference."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from time import perf_counter

from holdout import HERE, bundle


def measure(cache):
    reference = json.loads((HERE / "heldout.json").read_text())
    raw = (cache / "heldout.raw").read_bytes()
    previous = (cache / "heldout.pfw").read_bytes()
    basis = (cache / "heldout.base").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference["verified_sha256"]
    assert hashlib.sha256(previous).hexdigest() == reference["archive_sha256"]
    size = struct.unpack_from(">I", previous, 37)[0]
    manifest = json.loads(bundle.zstd(previous[41 : 41 + size], decode=True))
    records = bundle.unpack_rows(raw)
    results = {}
    for level in (12, 19, 22):
        start = perf_counter()
        packed, _ = bundle.encode(
            records, manifest, basis, strategy="columns", level=level
        )
        elapsed = perf_counter() - start
        assert bundle.decode(packed, lambda _: basis) == records
        if level == 19:
            assert packed == previous
        results[str(level)] = {
            "archive_bytes": len(packed),
            "archive_sha256": hashlib.sha256(packed).hexdigest(),
            "encode_seconds": round(elapsed, 3),
        }
    return {
        "records": len(records),
        "verified_sha256": reference["verified_sha256"],
        "zstd_version": subprocess.check_output(
            ["zstd", "--version"], text=True
        ).strip(),
        "profiles": results,
    }


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1])), indent=2))
