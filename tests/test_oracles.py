"""Sanity-check independent reference oracles in benchmarks/oracles.py."""

from __future__ import annotations

import math

import numpy as np

from benchmarks.oracles import (
    baseline_ewma_pandas,
    baseline_moments_pandas,
    baseline_moments_polars,
    baseline_moments_river,
    baseline_ridge_scipy,
    oracle_cusum_sequential,
    oracle_eprocess_sequential,
    oracle_ewma_numpy,
    oracle_kalman_cv_numpy,
    oracle_moments_numpy,
    oracle_ridge_numpy,
)


def test_moments_oracles_agree() -> None:
    rng = np.random.default_rng(1)
    xs = rng.normal(5.0, 2.5, size=500)
    m_np = oracle_moments_numpy(xs)
    m_pd = baseline_moments_pandas(xs)
    m_pl = baseline_moments_polars(xs)
    m_rv = baseline_moments_river(xs)
    for other in (m_pd, m_pl, m_rv):
        assert abs(m_np["mean"] - other["mean"]) < 1e-12
        assert abs(m_np["variance"] - other["variance"]) < 1e-11
        assert abs(m_np["min"] - other["min"]) < 1e-12
        assert abs(m_np["max"] - other["max"]) < 1e-12


def test_ewma_oracles_agree() -> None:
    rng = np.random.default_rng(2)
    ts = np.cumsum(rng.integers(1, 5, size=200)).astype(np.int64)
    xs = rng.normal(1.0, 3.0, size=200)
    half_life = 25.0
    lam = math.log(2.0) / half_life
    e_np = oracle_ewma_numpy(ts, xs, lam)
    e_pd = baseline_ewma_pandas(ts, xs, half_life)
    assert abs(e_np["mean"] - e_pd) < 1e-12


def test_cusum_and_eprocess_oracles() -> None:
    ts = np.arange(200, dtype=np.int64)
    xs = np.zeros(200, dtype=np.float64)
    xs[100:] = 2.0
    c = oracle_cusum_sequential(ts, xs, mu0=0.0, sigma0=1.0, slack_k=0.5, threshold=6.0)
    assert c["first_alarm_ts"] == 103
    assert c["max_peak"] == 100 * 1.5

    ep = oracle_eprocess_sequential(xs)
    assert ep["e_detector"] > 1e6
    assert ep["peak_e_detector_valid"] > 1e5


def test_kalman_and_ridge_oracles() -> None:
    rng = np.random.default_rng(3)
    ts = np.arange(300, dtype=np.int64)
    dts = np.ones(300, dtype=np.float64)
    ys = 0.5 * ts + rng.normal(0.0, 0.2, size=300)
    m, p = oracle_kalman_cv_numpy(dts, ys, 1e-3, 0.04, np.zeros(2), np.eye(2) * 10.0)
    assert abs(m[1] - 0.5) < 0.05
    assert p[0, 0] > 0.0 and p[1, 1] > 0.0

    xs = rng.normal(0.0, 1.0, size=(300, 4))
    y = xs @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(0.0, 0.1, size=300)
    w_np = oracle_ridge_numpy(ts, xs, y, math.log(2.0) / 100.0, 1e-2)
    w_sp = baseline_ridge_scipy(ts, xs, y, math.log(2.0) / 100.0, 1e-2)
    assert np.max(np.abs(w_np - w_sp)) < 1e-12
