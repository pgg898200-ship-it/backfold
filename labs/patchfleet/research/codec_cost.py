"""Measure held-out transfer and codec cost with the source dictionary already available."""

import json
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
from holdout import bundle


def run(work, output, fixed_only=False):
    base = work / "heldout.base"
    basis = base.read_bytes()
    raw = (work / "heldout.raw").read_bytes()
    records = bundle.unpack_rows(raw)
    archive = (work / "heldout.pfw").read_bytes()
    n = struct.unpack_from(">I", archive, 37)[0]
    manifest = json.loads(bundle.zstd(archive[41 : 41 + n], decode=True))

    def stock_encode():
        return subprocess.check_output(
            [
                "zstd",
                "-q",
                "--ultra",
                "-22",
                "--long=27",
                f"--patch-from={base}",
                f"--stream-size={len(raw)}",
                "-c",
            ],
            input=raw,
        )

    if fixed_only:
        times = []
        for _ in range(3):
            t = time.perf_counter()
            packed = bundle.encode(records, manifest, basis, strategy="columns")[0]
            times.append(time.perf_counter() - t)
            if packed != archive:
                raise ValueError("Fixed mode and automatic archive differ")
        output.write_text(
            json.dumps(
                dict(
                    runs=3,
                    median_encode_seconds=statistics.median(times),
                    individual_seconds=times,
                    archive_bytes=len(archive),
                ),
                indent=2,
            )
            + "\n"
        )
        return
    stock = stock_encode()

    def stock_decode():
        return subprocess.check_output(
            ["zstd", "-q", "-d", "--long=27", f"--patch-from={base}", "-c"], input=stock
        )

    funcs = {
        "stock_encode": stock_encode,
        "fleet_encode": lambda: bundle.encode(records, manifest, basis)[0],
        "stock_decode": stock_decode,
        "fleet_decode": lambda: bundle.pack_rows(
            bundle.decode(archive, lambda _: basis)
        ),
    }
    expect = {
        "stock_encode": stock,
        "fleet_encode": archive,
        "stock_decode": raw,
        "fleet_decode": raw,
    }
    times = {k: [] for k in funcs}
    for i in range(3):
        order = list(funcs)
        if i % 2:
            order.reverse()
        for k in order:
            t = time.perf_counter()
            r = funcs[k]()
            times[k].append(time.perf_counter() - t)
            if r != expect[k]:
                raise ValueError(k)
    output.write_text(
        json.dumps(
            dict(
                runs=3,
                median_seconds={k: statistics.median(v) for k, v in times.items()},
                individual_seconds=times,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), "--columns-only" in sys.argv[3:])
