"""Build and measure the C++17 interleaved-stream benchmark. Requires zlib."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--streams", type=int, default=2048)
parser.add_argument("--rounds", type=int, default=96)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
if not 1 <= args.repeats <= 100:
    parser.error("--repeats must be in 1..100")
binary = root / "build" / "fleet"
binary.parent.mkdir(exist_ok=True)
compiler = os.getenv("CXX", "c++")
subprocess.run([compiler, "-O3", "-std=c++17", "-Wall", "-Wextra", str(root / "scale/fleet.cpp"),
                str(root / "scale/sink.cpp"), "-lz", "-o", str(binary)], check=True)
cols = "codec streams rounds events raw_bytes wire_bytes encoder_state_bytes warm_ns_per_event".split()
runs = []
for trial in range(args.repeats):
    codecs = ["eventfold", "gzip-w12-m3"]
    if trial % 2:
        codecs.reverse()
    for codec in codecs:
        raw = subprocess.check_output([str(binary), codec, str(args.streams), str(args.rounds)], text=True)
        row = next(csv.DictReader(io.StringIO(raw), fieldnames=cols))
        runs.append({k: (v if k == "codec" else float(v) if k == "warm_ns_per_event" else int(v))
                     for k, v in row.items()})
summary = {c: statistics.median(r["warm_ns_per_event"] for r in runs if r["codec"] == c)
           for c in ["eventfold", "gzip-w12-m3"]}
print(json.dumps({"compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                  "system": platform.platform(), "median_warm_ns_per_event": summary, "runs": runs}, indent=2))
