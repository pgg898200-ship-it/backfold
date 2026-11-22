"""Comprehensive Python unit, parity, and adversarial tests for `backfold`."""

from __future__ import annotations

import math

import backfold as bf
import numpy as np
import pytest

from benchmarks.oracles import (
    oracle_cusum_sequential,
    oracle_eprocess_sequential,
    oracle_ewma_numpy,
    oracle_kalman_cv_numpy,
    oracle_moments_numpy,
    oracle_ridge_numpy,
)


def test_moments_out_of_order_retract_fork_and_high_offset() -> None:
    rng = np.random.default_rng(101)
    n = 500
    ts = np.arange(n, dtype=np.int64)
    # 1e12 offset cancellation torture
    xs = 1e12 + rng.normal(0.0, 2.5, size=n)

    ms = bf.MomentsStream()
    perm = rng.permutation(n)
    for i in perm.tolist():
        ms.upsert(int(ts[i]), float(xs[i]))

    assert len(ms) == n
    s = ms.summary()
    ref_centered = oracle_moments_numpy(xs - 1e12)
    assert abs(s["variance"] - ref_centered["variance"]) < 1e-10
    assert abs(s["skewness"] - ref_centered["skewness"]) < 1e-10
    assert abs(s["excess_kurtosis"] - ref_centered["excess_kurtosis"]) < 1e-10

    # Fork isolation + spike + retraction
    branch = ms.fork()
    branch.upsert(250, 1e12 + 1e6)
    assert branch.summary()["max"] > ms.summary()["max"] + 1e5
    assert abs(ms.summary()["variance"] - ref_centered["variance"]) < 1e-10

    assert branch.retract(250) is True
    assert branch.retract(250) is False
    branch.upsert(250, float(xs[250]))
    assert abs(branch.summary()["variance"] - ref_centered["variance"]) < 1e-10

    # Range query parity
    r = ms.range(100, 349)
    ref_r = oracle_moments_numpy(xs[100:350] - 1e12)
    assert r["n"] == 250
    assert abs(r["variance"] - ref_r["variance"]) < 1e-10


def test_ewma_irregular_timestamps_and_retraction() -> None:
    rng = np.random.default_rng(202)
    ts = np.cumsum(rng.integers(1, 50, size=400)).astype(np.int64)
    xs = rng.normal(3.0, 1.5, size=400)
    half_life = 120.0
    lam = math.log(2.0) / half_life

    es = bf.EwmaStream(half_life=half_life)
    es.ingest(ts, xs)
    ref = oracle_ewma_numpy(ts, xs, lam)
    got = es.summary()
    assert abs(got["mean"] - ref["mean"]) < 1e-12
    assert abs(got["variance"] - ref["variance"]) < 1e-12

    # Retract a point in the middle and verify exact parity
    drop = 177
    assert es.retract(int(ts[drop])) is True
    ts2 = np.delete(ts, drop)
    xs2 = np.delete(xs, drop)
    ref2 = oracle_ewma_numpy(ts2, xs2, lam)
    got2 = es.summary()
    assert abs(got2["mean"] - ref2["mean"]) < 1e-12
    assert abs(got2["variance"] - ref2["variance"]) < 1e-12


def test_cusum_exact_first_alarm_and_retroactive_shift() -> None:
    rng = np.random.default_rng(303)
    ts = np.arange(1_000, dtype=np.int64)
    zs = rng.normal(0.0, 1.0, size=1_000)
    zs[600:640] += 2.5

    cs = bf.CusumStream(mu0=0.0, sigma0=1.0, slack_k=0.5)
    cs.ingest(ts, zs)
    ref = oracle_cusum_sequential(ts, zs, 0.0, 1.0, 0.5, threshold=8.0)
    assert abs(cs.max_peak() - float(ref["max_peak"])) < 1e-11
    assert cs.first_alarm(8.0) == ref["first_alarm_ts"]

    # Insert an earlier late burst at t=150..158 -> first_alarm must move earlier!
    zs_mod = zs.copy()
    for t in range(150, 158):
        zs_mod[t] = 3.0
        cs.upsert(t, 3.0)
    ref2 = oracle_cusum_sequential(ts, zs_mod, 0.0, 1.0, 0.5, threshold=8.0)
    assert cs.first_alarm(8.0) == ref2["first_alarm_ts"]
    assert cs.first_alarm(8.0) is not None and cs.first_alarm(8.0) < 200


def test_kalman_1d_2d_and_3d_missing_obs_parity() -> None:
    rng = np.random.default_rng(404)
    n = 300
    ts = np.arange(n, dtype=np.int64)
    dts = rng.uniform(0.2, 2.5, size=n)
    ys = np.cumsum(rng.normal(0.05, 0.2, size=n)) + rng.normal(0.0, 0.4, size=n)
    m0 = np.array([0.2, -0.1])
    p0 = np.array([[4.0, 0.2], [0.2, 1.5]])

    kf = bf.KalmanStream(dim=2)
    kf.ingest_constant_velocity(ts, ys, dts, q_accel=0.05, r_obs=0.3)
    bm, bp = kf.posterior(m0.tolist(), p0.ravel().tolist())
    om, op = oracle_kalman_cv_numpy(dts, ys, 0.05, 0.3, m0, p0)
    assert np.max(np.abs(np.array(bm) - om)) < 1e-11
    assert np.max(np.abs(np.array(bp).reshape(2, 2) - op)) < 1e-11

    # Retract t=120 and replace with missing prediction step
    f_miss = [1.0, float(dts[120]), 0.0, 1.0]
    dt = float(dts[120])
    q_miss = [0.05 * dt**3 / 3.0, 0.05 * dt**2 / 2.0, 0.05 * dt**2 / 2.0, 0.05 * dt]
    kf.upsert_missing(120, f_miss, q_miss)
    bm2, bp2 = kf.posterior(m0.tolist(), p0.ravel().tolist())
    assert all(math.isfinite(v) for v in bm2 + bp2)
    assert len(kf) == n


def test_ridge_discounted_and_ill_conditioned_parity() -> None:
    rng = np.random.default_rng(505)
    n, d = 600, 8
    ts = np.arange(n, dtype=np.int64)
    xmat = rng.normal(0.0, 1.0, size=(n, d))
    xmat[:, 0] *= 1e4
    xmat[:, 1] *= 1e-4
    w_true = np.array([1e-4, 1e4, 0.5, -0.3, 0.2, -0.7, 0.1, 0.9])
    yvec = xmat @ w_true + rng.normal(0.0, 0.05, size=n)
    lam = math.log(2.0) / 300.0
    alpha = 1e-6

    rs = bf.RidgeStream(dim=d, decay_rate=lam)
    rs.ingest(ts, xmat, yvec)
    sol = rs.solve(alpha)
    ow = oracle_ridge_numpy(ts, xmat, yvec, lam, alpha)
    assert np.max(np.abs(np.array(sol["weights"]) - ow)) < 1e-6

    y_hat, pvar = rs.predict_with_variance(xmat[-1].tolist(), alpha)
    assert math.isfinite(y_hat) and pvar > 0.0


def test_eprocess_ebh_and_swarm_echo_dedup() -> None:
    rng = np.random.default_rng(606)
    zs = rng.normal(0.0, 1.0, size=300)
    ep = bf.EProcessStream()
    ep.ingest_z(np.arange(300, dtype=np.int64), zs)
    oe = oracle_eprocess_sequential(zs)
    assert abs(ep.log_wealth() - oe["log_wealth"]) < 1e-11
    assert abs(ep.log_e_detector() - oe["log_e_detector"]) < 1e-11
    assert ep.e_value() > 0.0

    # Swarm echo deduplication
    ledger = bf.SwarmEProcessLedger()
    for a in range(4):
        ledger.spawn_agent(a)
    for i in range(100):
        assert ledger.observe_z(0, i, 1, 0, float(zs[i])) is True
        assert ledger.observe_z(1, i, 1, 0, float(zs[i])) is False
        assert ledger.observe_z(2, i, 1, 0, float(zs[i])) is False
    merged = ledger.merge_all()
    st = ledger.stats()
    assert st["raw_observations"] == 300
    assert st["unique_events"] == 100
    assert st["echoes_deduped"] == 200
    assert st["overwrites"] == 0
    assert math.isfinite(merged["e_value"])


def test_bitemporal_zero_lookahead_and_nan_rejection() -> None:
    bt = bf.BitemporalMoments()
    bt.ingest_event(event_time=10, knowledge_time=10, x=2.0)
    bt.ingest_event(event_time=20, knowledge_time=20, x=4.0)
    # Late restatement of t=10 arriving at knowledge_time=50
    bt.ingest_event(event_time=10, knowledge_time=50, x=100.0)

    s25 = bt.as_of(25)
    assert s25["n"] == 2 and abs(s25["mean"] - 3.0) < 1e-12
    s50 = bt.as_of(50)
    assert s50["n"] == 2 and abs(s50["mean"] - 52.0) < 1e-12

    ms = bf.MomentsStream()
    with pytest.raises(ValueError):
        ms.upsert(1, float("nan"))
    with pytest.raises(ValueError):
        ms.upsert(1, float("inf"))


def test_digest_stream_catches_burst_and_compresses() -> None:
    rng = np.random.default_rng(707)
    n = 10_000
    ts = np.arange(n, dtype=np.int64)
    xs = rng.normal(0.0, 1.0, size=n)
    xs[4200:4205] += 9.0
    xs[7000:] += 2.0

    dig = bf.digest_stream("test_ch", ts, xs, top_k=4, threshold=18.0)
    assert dig["compression_ratio"] > 100.0
    assert dig["digest_approx_tokens"] < 350
    assert dig["first_alarm_ts"] is not None and 4198 <= dig["first_alarm_ts"] <= 4210
    assert any(w["t_start"] <= 4202 <= w["t_end"] for w in dig["hotspots"])
    assert "# Stream Digest:" in dig["markdown"]


def test_kalman_rts_smoother_and_marginal_log_likelihood() -> None:
    from benchmarks.oracles import oracle_kalman_rts_cv_numpy

    rng = np.random.default_rng(808)
    n = 120
    ts = np.arange(n, dtype=np.int64)
    dts = rng.uniform(0.3, 1.8, size=n)
    ys = np.cumsum(rng.normal(0.08, 0.15, size=n)) + rng.normal(0.0, 0.4, size=n)
    # Mark several missing observations
    ys[17] = float("nan")
    ys[63] = float("nan")
    m0 = np.array([0.3, -0.1])
    p0 = np.array([[5.0, 0.1], [0.1, 2.0]])

    kf = bf.KalmanStream(dim=2)
    for t, dt, y in zip(ts.tolist(), dts.tolist(), ys.tolist()):
        if math.isnan(y):
            f_miss = [1.0, dt, 0.0, 1.0]
            q_miss = [0.04 * dt**3 / 3.0, 0.04 * dt**2 / 2.0, 0.04 * dt**2 / 2.0, 0.04 * dt]
            kf.upsert_missing(t, f_miss, q_miss)
        else:
            kf.upsert_constant_velocity(t, y, dt, 0.04, 0.35)

    # Late edit at t = 40
    ys[40] += 3.5
    kf.upsert_constant_velocity(40, float(ys[40]), float(dts[40]), 0.04, 0.35)

    m0_l = m0.tolist()
    p0_l = p0.ravel().tolist()
    got_ll = kf.log_marginal_likelihood(m0_l, p0_l)

    query_pts = [0, 17, 40, 60, 63, 119]
    batch_means, batch_covs = kf.smooth_many(np.array(query_pts, dtype=np.int64), m0_l, p0_l)
    for idx, q in enumerate(query_pts):
        om, op, oll = oracle_kalman_rts_cv_numpy(dts, ys, 0.04, 0.35, m0, p0, q)
        if idx == 0:
            assert abs(got_ll - oll) < 1e-10
        sm_m, sm_p = kf.smooth_at(q, m0_l, p0_l)
        assert np.max(np.abs(np.array(sm_m) - om)) < 1e-10
        assert np.max(np.abs(np.array(sm_p).reshape(2, 2) - op)) < 1e-10
        assert np.max(np.abs(np.array(batch_means[idx]) - om)) < 1e-10
        assert np.max(np.abs(np.array(batch_covs[idx]).reshape(2, 2) - op)) < 1e-10

    # Range marginal log-likelihood on a sub-window [20, 79]
    sub_ll = kf.range_log_marginal_likelihood(20, 79, m0_l, p0_l)
    _, _, sub_oll = oracle_kalman_rts_cv_numpy(
        dts[20:80], ys[20:80], 0.04, 0.35, m0, p0, 59
    )
    assert abs(sub_ll - sub_oll) < 1e-10


def test_covariance_stream_high_offset_range_and_mahalanobis() -> None:
    from benchmarks.oracles import oracle_covariance_numpy

    rng = np.random.default_rng(909)
    base = 1e12
    for dim in (5, 10):
        n = 300
        ts = np.arange(n, dtype=np.int64)
        centered = rng.normal(0.0, 2.0, size=(n, dim))
        centered[:, 1] = 0.8 * centered[:, 0] + 0.4 * centered[:, 1]
        xmat = base + centered
        exact_centered = xmat - base

        cs = bf.CovarianceStream(dim=dim)
        cs.ingest(ts, xmat)
        assert len(cs) == n

        got_cov = np.array(cs.covariance(ddof=1))
        ref_cov = oracle_covariance_numpy(exact_centered, ddof=1)
        assert np.max(np.abs(got_cov - ref_cov)) < 1e-10

        got_corr = np.array(cs.correlation())
        ref_corr = np.corrcoef(exact_centered, rowvar=False)
        assert np.max(np.abs(got_corr - ref_corr)) < 1e-10

        # Range query [50, 200]
        got_rcov = np.array(cs.range_covariance(50, 200, ddof=1))
        ref_rcov = oracle_covariance_numpy(exact_centered[50:201], ddof=1)
        assert np.max(np.abs(got_rcov - ref_rcov)) < 1e-10

        # Fork + spike + retract isolation
        branch = cs.fork()
        branch.upsert(120, [base + 1e6] * dim)
        assert np.max(np.abs(np.array(cs.covariance(1)) - ref_cov)) < 1e-10
        assert branch.retract(120) is True
        branch.upsert(120, xmat[120].tolist())
        assert np.max(np.abs(np.array(branch.covariance(1)) - ref_cov)) < 1e-10

        # Single vs batch Mahalanobis parity + outlier discrimination
        qmat = base + rng.normal(0.0, 2.0, size=(8, dim))
        qmat[-1] = base + 40.0
        batch = cs.mahalanobis_many(qmat, ridge_eps=1e-5)
        for i in range(8):
            single = cs.mahalanobis_sq(qmat[i].tolist(), ridge_eps=1e-5)
            assert abs(single - batch[i]) < 1e-11
        assert batch[-1] > 100.0

    cs_err = bf.CovarianceStream(dim=3)
    with pytest.raises(ValueError):
        cs_err.upsert(0, [1.0, float("nan"), 2.0])

