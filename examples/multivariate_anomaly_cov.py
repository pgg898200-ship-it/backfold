"""Exact O(log T) Multivariate Covariance, Correlation & Mahalanobis Anomaly Demo.

Demonstrates:
1. High-offset (1e12) multivariate telemetry with zero catastrophic cancellation (106-bit TwoSum mean).
2. O(log T) late-arrival repair & toxic-outlier retraction without full recomputation.
3. Exact sub-window regime correlation via `range_correlation(t_start, t_end)`.
4. Factored batch Mahalanobis anomaly scoring via `mahalanobis_many`.
"""

from __future__ import annotations

import time

import backfold as bf
import numpy as np


def main() -> None:
    rng = np.random.default_rng(7)
    n, dim = 50_000, 8
    ts = np.arange(n, dtype=np.int64)
    base = 1e12  # Catastrophic for naive E[X^2] - E[X]^2 formulas!

    # Correlated 8-sensor stream (sensors 0 and 1 strongly coupled in regime 1, decoupled in regime 2)
    z = rng.normal(0.0, 1.0, size=(n, dim))
    z[:25_000, 1] = 0.9 * z[:25_000, 0] + 0.3 * z[:25_000, 1]
    z[25_000:, 1] = -0.8 * z[25_000:, 0] + 0.4 * z[25_000:, 1]
    xmat = base + z

    cs = bf.CovarianceStream(dim=dim)
    cs.ingest(ts, xmat)

    # Inject a massive toxic sensor glitch at t = 12,345, then retract it cleanly
    cs.upsert(12_345, [base + 1e9] * dim)
    t0 = time.perf_counter()
    cs.retract(12_345)
    cs.upsert(12_345, xmat[12_345].tolist())
    r1 = cs.range_correlation(0, 24_999)
    r2 = cs.range_correlation(25_000, n - 1)
    repair_us = (time.perf_counter() - t0) * 1e6

    print(f"T={n:,}, dim={dim} (offset=1e12): retract + repair + 2 regime correlations in {repair_us:.1f} us")
    print(f"  Regime 1 corr(s0, s1) = {r1[0][1]:+.4f} (expected ~+0.95)")
    print(f"  Regime 2 corr(s0, s1) = {r2[0][1]:+.4f} (expected ~-0.89)")
    assert r1[0][1] > 0.90 and r2[0][1] < -0.80

    # Batch Mahalanobis anomaly scoring against the clean stream covariance
    probes = np.vstack([base + rng.normal(0.0, 1.0, size=(4, dim)), base + np.full((1, dim), 12.0)])
    d2_scores = cs.mahalanobis_many(probes, ridge_eps=1e-6)
    print(f"  Normal probes Mahalanobis^2: {[round(s, 2) for s in d2_scores[:-1]]}")
    print(f"  Anomalous probe Mahalanobis^2: {d2_scores[-1]:.1f} (clearly flagged!)")
    assert d2_scores[-1] > 500.0


if __name__ == "__main__":
    main()
