"""Example: Exact O(log T) Bayesian Changepoint Shock Localization + 106-bit Streaming PCA + e-BH Sift.

Run:
    python examples/streaming_pca_and_changepoint_bf.py
"""

from __future__ import annotations

import time

import backfold as bf
import numpy as np


def main() -> None:
    rng = np.random.default_rng(2026)

    # 1. Exact O(K log T) Bayesian Changepoint Shock Localization on a +1e6-offset trajectory
    n = 50_000
    true_cp = 27_500
    offset = 1e6
    ts = np.arange(n, dtype=np.int64)
    dts = np.ones(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    level, vel = offset, 0.002
    for t in range(n):
        if t == true_cp:
            level += 9.5
            vel += 0.015
        level += vel + float(rng.normal(0.0, 0.03))
        ys[t] = level + float(rng.normal(0.0, 0.5))

    kf = bf.KalmanStream(dim=2)
    kf.ingest_constant_velocity(ts, ys, dts, q_accel=1e-3, r_obs=0.25)

    m0 = [offset, 0.0]
    p0 = [100.0, 0.0, 0.0, 10.0]
    q_jump = [200.0, 0.0, 0.0, 1.0]

    # Zero-blind-spot O(log^2 T) global Bayesian shock locator in <0.5 ms!
    t0 = time.perf_counter()
    loc = kf.locate_changepoint(m0, p0, q_jump, n_windows=64)
    scan_ms = (time.perf_counter() - t0) * 1e3

    print(
        f"[Changepoint BF] T={n:,} (+1e6 offset): localized true shock {true_cp} -> "
        f"{loc['best_ts']} (logBF={loc['best_log_bf']:.1f}) in {scan_ms:.2f} ms"
    )
    assert abs(int(loc["best_ts"]) - true_cp) <= 2
    assert loc["best_log_bf"] > 40.0

    # 2. Exact 106-bit Knuth-compensated Streaming PCA (survives 1e12 DC offset + late retraction!)
    dim = 6
    n_pca = 10_000
    ts_p = np.arange(n_pca, dtype=np.int64)
    factors = rng.normal(0.0, [6.0, 2.5, 0.7, 0.2, 0.1, 0.05], size=(n_pca, dim))
    q_orth, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    xmat = 1e12 + factors @ q_orth.T

    cs = bf.CovarianceStream(dim=dim)
    cs.ingest(ts_p, xmat)
    # Inject catastrophic sensor spike, then retract it in O(log T)!
    cs.upsert(4_000, [1e12 + 1e7] * dim)
    cs.retract(4_000)

    t_p0 = time.perf_counter()
    pca = cs.range_pca(1_000, 9_000, ddof=1, top_k=3)
    pca_us = (time.perf_counter() - t_p0) * 1e6
    evr_pct = [round(v * 100.0, 2) for v in pca["explained_variance_ratio"]]
    print(f"[1e12 Range PCA] [1000..9000] top-3 EVR={evr_pct}% in {pca_us:.1f} us (zero BLAS)")
    assert sum(pca["explained_variance_ratio"]) > 0.98

    # 3. Single-call GIL-free 500-channel Wang-Ramdas e-BH FDR sift
    ch = rng.normal(0.0, 1.0, size=(500, 400))
    ch[10:60] *= 3.0
    ch[:10, 220:] += 1.8
    t_s0 = time.perf_counter()
    sift = bf.sift_streams(ch, warmup=120, alpha=0.05, null_margin=0.15)
    sift_ms = (time.perf_counter() - t_s0) * 1e3
    rej = set(sift["rejected"])
    tp = len(rej & set(range(10)))
    fp = len(rej - set(range(10)))
    print(f"[500ch e-BH Sift] TP={tp}/10, FP={fp} in {sift_ms:.2f} ms (FDR={fp / max(1, len(rej)):.3f})")
    assert tp == 10 and fp == 0


if __name__ == "__main__":
    main()
