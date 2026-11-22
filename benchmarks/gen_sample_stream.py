"""Generate reproducible synthetic multi-regime telemetry CSV for CLI digest/drill tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_stream_csv(out_path: Path, n: int = 100_000, seed: int = 2026) -> None:
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64)
    # Base stationary noise
    vals = rng.normal(0.0, 1.0, size=n)
    # Plant 1: Regime shift at t=40,000 (+1.8 sigma sustained)
    vals[40_000:70_000] += 1.8
    # Plant 2: Recovery + negative shift at t=70,000 (-1.2 sigma)
    vals[70_000:] -= 1.2
    # Plant 3: Rare acute burst (+8.5 sigma) at t=22,500..22,505
    vals[22_500:22_505] += 8.5
    # Plant 4: Acute drop (-7.5 sigma) at t=88,100..88,103
    vals[88_100:88_103] -= 7.5

    lines = ["ts,value\n"]
    lines.extend(f"{int(t)},{v:.6f}\n" for t, v in zip(ts, vals))
    out_path.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("benchmarks/sample_100k.csv"))
    parser.add_argument("--n", type=int, default=100_000)
    args = parser.parse_args()
    generate_stream_csv(args.out, n=args.n)
    print(f"Wrote {args.n} rows to {args.out}")
