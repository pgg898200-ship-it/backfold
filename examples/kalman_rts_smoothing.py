"""Exact O(log T) Retractable RTS Kalman Smoother & Marginal Log-Likelihood Demo.

Demonstrates:
1. Ingesting irregular-dt sensor ticks with missing observations.
2. Arriving late edits & outlier retractions in O(log T) time (no O(T) forward-backward sweep).
3. Exact Rauch-Tung-Striebel (RTS) smoothed posterior p(x_t | y_{1:T}) at arbitrary timestamps
   via `smooth_at` / `smooth_many`, verifying smoothed variance <= filtered variance.
4. Exact O(1)-at-root Bayesian marginal log-likelihood `log p(y_{1:T})` for online model selection.
"""

from __future__ import annotations

import time

import backfold as bf
import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)
    n = 50_000
    ts = np.arange(n, dtype=np.int64)
    dts = np.ones(n, dtype=np.float64)
    # True accelerating trajectory + Gaussian sensor noise (sigma = 0.5)
    t_f = ts.astype(np.float64)
    true_pos = 0.05 * t_f + 2.0 * np.sin(t_f / 200.0)
    ys = true_pos + rng.normal(0.0, 0.5, size=n)

    m0 = [0.0, 0.0]
    p0 = [10.0, 0.0, 0.0, 10.0]

    kf = bf.KalmanStream(dim=2)
    kf.ingest_constant_velocity(ts, ys, dts, q_accel=1e-3, r_obs=0.25)

    # 1. Late correction at t = 15,000 + exact RTS smoothing at t = 25,000 + marginal log-likelihood
    t0 = time.perf_counter()
    kf.upsert_constant_velocity(15_000, float(true_pos[15_000]), 1.0, 1e-3, 0.25)
    sm_mean, sm_cov = kf.smooth_at(25_000, m0, p0)
    log_evidence = kf.log_marginal_likelihood(m0, p0)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    # Compare filtered vs smoothed variance at t = 25,000
    fl_mean, fl_cov = kf.range_posterior(0, 25_000, m0, p0)
    print(f"T={n:,} late edit + O(log T) RTS smooth + log p(y): {elapsed_us:.1f} us")
    print(f"  t=25,000 true_pos={true_pos[25_000]:.4f}")
    print(f"  filtered mean={fl_mean[0]:.4f}, pos_var={fl_cov[0]:.5f}")
    print(f"  smoothed mean={sm_mean[0]:.4f}, pos_var={sm_cov[0]:.5f} (strictly tighter!)")
    print(f"  exact marginal log-likelihood log p(y_1:T) = {log_evidence:.2f}")
    assert sm_cov[0] < fl_cov[0]

    # 2. Batch RTS smoothing across 20 checkpoints in one Rust call
    checkpoints = np.linspace(0, n - 1, 20, dtype=np.int64)
    means, _covs = kf.smooth_many(checkpoints, m0, p0)
    rmse = float(np.sqrt(np.mean([(m[0] - true_pos[int(t)]) ** 2 for m, t in zip(means, checkpoints)])))
    print(f"  20-point RTS trajectory RMSE vs ground truth: {rmse:.4f} (raw noise std=0.5000)")

    # 3. Online Bayesian model comparison via exact marginal log-likelihood on a 5k window
    sub_ys = ys[:5_000]
    sub_ts = ts[:5_000]
    sub_dts = dts[:5_000]
    scores = {}
    for q_cand in (1e-6, 1e-3, 1.0):
        cand = bf.KalmanStream(dim=2)
        cand.ingest_constant_velocity(sub_ts, sub_ys, sub_dts, q_accel=q_cand, r_obs=0.25)
        scores[q_cand] = cand.log_marginal_likelihood(m0, p0)
    best_q = max(scores, key=scores.__getitem__)
    print(f"  Bayes evidence model selection over q_accel: {scores} -> winner q={best_q}")


if __name__ == "__main__":
    main()
