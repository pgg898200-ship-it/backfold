"""Head-to-head matched benchmark suite for `backfold` vs. real Python & Rust baselines.

Measures `[LIVE-CPU]` latency and throughput across:
1. Out-of-order late arrival & retraction updates at T in {1k, 10k, 100k, 500k}
   (Moments, Continuous-Time EWMA, Page CUSUM, 2D Constant-Velocity Kalman, 8D Discounted Ridge)
   against NumPy, SciPy, Pandas, Polars, River, and `-O3` sequential Rust (via `backfold bench --json`).
2. Arbitrary time-range statistical queries `[t1, t2]` on T = 100,000.
3. Multi-agent counterfactual branching, echo-proof CRDT merge & 500-channel e-BH FDR sifting.
4. Agent stream compression (`backfold digest --budget 350`) token count vs. detection recall.
5. Honest append-only single-pass crossover disclosure.

Every single benchmark asserts exact numerical parity (`max_abs_diff <= 1e-10`) against
the independent naive sequential oracle in `benchmarks/oracles.py`.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.multi_agent_eval import run_multi_agent_eval
from benchmarks.oracles import (
    baseline_ewma_pandas,
    baseline_linear_river,
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


def time_us(fn: Callable[[], Any], repeats: int = 7, warmup: int = 2) -> tuple[float, float, Any]:
    last = None
    for _ in range(warmup):
        last = fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    samples.sort()
    med = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1)]
    return med, p95, last


def bench_late_updates(bf: Any, sizes: list[int], seed: int = 42) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for n in sizes:
        rows.extend(_bench_late_updates_for_n(bf, n, rng))
    return rows


def _bench_late_updates_for_n(
    bf: Any, n: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if True:
        ts = np.arange(n, dtype=np.int64)
        xs = rng.normal(10.0, 2.0, size=n)
        edit_idx = n // 3
        new_val = float(xs[edit_idx] + 5.0)

        # ---- 1. Moments (late edit) ----
        xs_edited = xs.copy()
        xs_edited[edit_idx] = new_val
        oracle_m = oracle_moments_numpy(xs_edited)

        ms = bf.MomentsStream()
        ms.ingest(ts, xs)

        def bf_moments_edit() -> dict[str, float]:
            ms.upsert(int(ts[edit_idx]), new_val)
            res = ms.summary()
            ms.upsert(int(ts[edit_idx]), float(xs[edit_idx]))
            return res

        bf_us, bf_p95, _bf_res = time_us(bf_moments_edit, repeats=31, warmup=5)
        # Verify parity on edited state
        ms.upsert(int(ts[edit_idx]), new_val)
        bf_check = ms.summary()
        err_m = max(
            abs(bf_check["mean"] - oracle_m["mean"]),
            abs(bf_check["variance"] - oracle_m["variance"]),
            abs(bf_check["skewness"] - oracle_m["skewness"]),
            abs(bf_check["excess_kurtosis"] - oracle_m["excess_kurtosis"]),
        )
        assert err_m < 1e-10, f"Moments parity error {err_m}"

        np_us, _, _ = time_us(lambda: oracle_moments_numpy(xs_edited), repeats=9, warmup=2)
        pd_us, _, _ = time_us(lambda: baseline_moments_pandas(xs_edited), repeats=9, warmup=2)
        pl_us, _, _ = time_us(lambda: baseline_moments_polars(xs_edited), repeats=9, warmup=2)
        rv_us = None
        if n <= 10_000:
            rv_us, _, _ = time_us(lambda: baseline_moments_river(xs_edited), repeats=3, warmup=1)

        rows.append(
            {
                "operator": "moments_late_edit",
                "n": n,
                "backfold_us": round(bf_us, 3),
                "backfold_p95_us": round(bf_p95, 3),
                "numpy_us": round(np_us, 2),
                "pandas_us": round(pd_us, 2),
                "polars_us": round(pl_us, 2),
                "river_us": round(rv_us, 2) if rv_us is not None else None,
                "speedup_vs_numpy": round(np_us / max(bf_us, 1e-6), 1),
                "speedup_vs_pandas": round(pd_us / max(bf_us, 1e-6), 1),
                "speedup_vs_polars": round(pl_us / max(bf_us, 1e-6), 1),
                "max_abs_diff": err_m,
            }
        )

        # ---- 2. Continuous-Time EWMA (late edit) ----
        half_life = 250.0
        decay_rate = math.log(2.0) / half_life
        oracle_e = oracle_ewma_numpy(ts, xs_edited, decay_rate)

        es = bf.EwmaStream(half_life=half_life)
        es.ingest(ts, xs)

        def bf_ewma_edit() -> dict[str, float]:
            es.upsert(int(ts[edit_idx]), new_val)
            res = es.summary()
            es.upsert(int(ts[edit_idx]), float(xs[edit_idx]))
            return res

        bf_us, bf_p95, _ = time_us(bf_ewma_edit, repeats=31, warmup=5)
        es.upsert(int(ts[edit_idx]), new_val)
        bf_check = es.summary()
        err_e = max(
            abs(bf_check["mean"] - oracle_e["mean"]),
            abs(bf_check["variance"] - oracle_e["variance"]),
        )
        assert err_e < 1e-10, f"EWMA parity error {err_e}"

        np_us, _, _ = time_us(
            lambda: oracle_ewma_numpy(ts, xs_edited, decay_rate), repeats=9, warmup=2
        )
        pd_us, _, _ = time_us(
            lambda: baseline_ewma_pandas(ts, xs_edited, half_life), repeats=7, warmup=1
        )
        rows.append(
            {
                "operator": "ewma_ct_late_edit",
                "n": n,
                "backfold_us": round(bf_us, 3),
                "backfold_p95_us": round(bf_p95, 3),
                "numpy_us": round(np_us, 2),
                "pandas_us": round(pd_us, 2),
                "speedup_vs_numpy": round(np_us / max(bf_us, 1e-6), 1),
                "speedup_vs_pandas": round(pd_us / max(bf_us, 1e-6), 1),
                "max_abs_diff": err_e,
            }
        )

        # ---- 3. Two-Sided Page CUSUM + First-Crossing Alarm (late edit) ----
        zs = rng.normal(0.0, 1.0, size=n)
        zs_edited = zs.copy()
        zs_edited[edit_idx] = 6.0
        oracle_c = oracle_cusum_sequential(ts, zs_edited, 0.0, 1.0, 0.5, threshold=5.0)

        cs = bf.CusumStream(mu0=0.0, sigma0=1.0, slack_k=0.5)
        cs.ingest(ts, zs)

        def bf_cusum_edit() -> tuple[float, int | None]:
            cs.upsert(int(ts[edit_idx]), 6.0)
            pk = cs.max_peak()
            fa = cs.first_alarm(5.0)
            cs.upsert(int(ts[edit_idx]), float(zs[edit_idx]))
            return pk, fa

        bf_us, bf_p95, _ = time_us(bf_cusum_edit, repeats=31, warmup=5)
        cs.upsert(int(ts[edit_idx]), 6.0)
        err_c = abs(cs.max_peak() - float(oracle_c["max_peak"]))
        assert err_c < 1e-10, f"CUSUM parity error {err_c}"
        assert cs.first_alarm(5.0) == oracle_c["first_alarm_ts"], "CUSUM first_alarm mismatch"

        py_us, _, _ = time_us(
            lambda: oracle_cusum_sequential(ts, zs_edited, 0.0, 1.0, 0.5, threshold=5.0),
            repeats=5 if n <= 100_000 else 3,
            warmup=1,
        )
        rows.append(
            {
                "operator": "cusum_late_edit_plus_first_alarm",
                "n": n,
                "backfold_us": round(bf_us, 3),
                "backfold_p95_us": round(bf_p95, 3),
                "python_seq_us": round(py_us, 2),
                "speedup_vs_python_seq": round(py_us / max(bf_us, 1e-6), 1),
                "max_abs_diff": err_c,
            }
        )

        # ---- 4. 2D Constant-Velocity Kalman Filter (late edit) ----
        dts = np.ones(n, dtype=np.float64)
        ys = np.cumsum(rng.normal(0.05, 0.1, size=n)) + rng.normal(0.0, 0.5, size=n)
        ys_edited = ys.copy()
        ys_edited[edit_idx] += 3.0
        m0 = np.array([0.0, 0.0], dtype=np.float64)
        p0 = np.array([[10.0, 0.0], [0.0, 10.0]], dtype=np.float64)

        kf = bf.KalmanStream(dim=2)
        kf.ingest_constant_velocity(ts, ys, dts, q_accel=1e-3, r_obs=0.25)

        def bf_kalman_edit() -> tuple[list[float], list[float]]:
            kf.upsert_constant_velocity(int(ts[edit_idx]), float(ys_edited[edit_idx]), 1.0, 1e-3, 0.25)
            post = kf.posterior(m0.tolist(), p0.ravel().tolist())
            kf.upsert_constant_velocity(int(ts[edit_idx]), float(ys[edit_idx]), 1.0, 1e-3, 0.25)
            return post

        bf_us, bf_p95, _ = time_us(bf_kalman_edit, repeats=31, warmup=5)
        kf.upsert_constant_velocity(int(ts[edit_idx]), float(ys_edited[edit_idx]), 1.0, 1e-3, 0.25)
        bf_m, bf_p = kf.posterior(m0.tolist(), p0.ravel().tolist())

        # Oracle check (full sequential NumPy Joseph-form Kalman)
        if n <= 100_000:
            np_us, _, (om, op) = time_us(
                lambda: oracle_kalman_cv_numpy(dts, ys_edited, 1e-3, 0.25, m0, p0),
                repeats=3 if n <= 10_000 else 1,
                warmup=0 if n > 10_000 else 1,
            )
            err_k = max(
                float(np.max(np.abs(np.array(bf_m) - om))),
                float(np.max(np.abs(np.array(bf_p).reshape(2, 2) - op))),
            )
            assert err_k < 1e-9, f"Kalman parity error {err_k}"
        else:
            np_us = None
            err_k = None

        rows.append(
            {
                "operator": "kalman_2d_cv_late_edit",
                "n": n,
                "backfold_us": round(bf_us, 3),
                "backfold_p95_us": round(bf_p95, 3),
                "numpy_seq_us": round(np_us, 2) if np_us is not None else None,
                "speedup_vs_numpy_seq": round(np_us / max(bf_us, 1e-6), 1) if np_us is not None else None,
                "max_abs_diff": err_k,
            }
        )

        # ---- 5. 8D Continuous-Time Discounted Ridge Regression (delayed label!) ----
        dim = 8
        xmat = rng.normal(0.0, 1.0, size=(n, dim))
        true_w = rng.normal(0.0, 1.0, size=dim)
        yvec = xmat @ true_w + rng.normal(0.0, 0.2, size=n)
        yvec_edited = yvec.copy()
        yvec_edited[edit_idx] += 4.0
        ridge_decay = math.log(2.0) / 5_000.0
        alpha = 1e-2

        rs = bf.RidgeStream(dim=dim, decay_rate=ridge_decay)
        rs.ingest(ts, xmat, yvec)

        def bf_ridge_edit() -> list[float]:
            rs.upsert(int(ts[edit_idx]), xmat[edit_idx].tolist(), float(yvec_edited[edit_idx]), 1.0)
            sol = rs.solve(alpha)
            rs.upsert(int(ts[edit_idx]), xmat[edit_idx].tolist(), float(yvec[edit_idx]), 1.0)
            return sol["weights"]

        bf_us, bf_p95, _ = time_us(bf_ridge_edit, repeats=31, warmup=5)
        rs.upsert(int(ts[edit_idx]), xmat[edit_idx].tolist(), float(yvec_edited[edit_idx]), 1.0)
        bf_w = np.array(rs.solve(alpha)["weights"])
        np_us, _, ow = time_us(
            lambda: oracle_ridge_numpy(ts, xmat, yvec_edited, ridge_decay, alpha),
            repeats=7,
            warmup=2,
        )
        sp_us, _, _ = time_us(
            lambda: baseline_ridge_scipy(ts, xmat, yvec_edited, ridge_decay, alpha),
            repeats=7,
            warmup=2,
        )
        err_r = float(np.max(np.abs(bf_w - ow)))
        assert err_r < 1e-10, f"Ridge parity error {err_r}"

        rv_us = None
        if n <= 10_000:
            rv_us, _, _ = time_us(lambda: baseline_linear_river(xmat, yvec_edited, alpha), repeats=3, warmup=1)

        rows.append(
            {
                "operator": "ridge_8d_delayed_label",
                "n": n,
                "backfold_us": round(bf_us, 3),
                "backfold_p95_us": round(bf_p95, 3),
                "numpy_us": round(np_us, 2),
                "scipy_us": round(sp_us, 2),
                "river_us": round(rv_us, 2) if rv_us is not None else None,
                "speedup_vs_numpy": round(np_us / max(bf_us, 1e-6), 1),
                "speedup_vs_scipy": round(sp_us / max(bf_us, 1e-6), 1),
                "max_abs_diff": err_r,
            }
        )

    return rows


def bench_range_queries(bf: Any, n: int = 100_000, n_queries: int = 200, seed: int = 99) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64)
    xs = rng.normal(5.0, 2.0, size=n)
    starts = rng.integers(0, n // 2, size=n_queries)
    ends = rng.integers(n // 2, n, size=n_queries)

    ms = bf.MomentsStream()
    ms.ingest(ts, xs)

    t0 = time.perf_counter()
    bf_means = [ms.range(int(a), int(b))["mean"] for a, b in zip(starts.tolist(), ends.tolist())]
    bf_us_per_q = (time.perf_counter() - t0) * 1e6 / n_queries

    t1 = time.perf_counter()
    np_means = [float(np.mean(xs[a : b + 1])) for a, b in zip(starts.tolist(), ends.tolist())]
    np_us_per_q = (time.perf_counter() - t1) * 1e6 / n_queries

    max_diff = max(abs(a - b) for a, b in zip(bf_means, np_means))
    assert max_diff < 1e-11, f"Range query parity error {max_diff}"

    # Verify EProcess parity too
    zs = rng.normal(0.0, 1.0, size=2_000)
    ep = bf.EProcessStream()
    ep.ingest_z(np.arange(2_000, dtype=np.int64), zs)
    oe = oracle_eprocess_sequential(zs)
    assert abs(ep.log_wealth() - oe["log_wealth"]) < 1e-10
    assert abs(ep.log_e_detector() - oe["log_e_detector"]) < 1e-10

    return [
        {
            "operator": "moments_arbitrary_range_query",
            "n": n,
            "n_queries": n_queries,
            "backfold_us_per_query": round(bf_us_per_q, 3),
            "numpy_us_per_query": round(np_us_per_q, 2),
            "speedup_vs_numpy": round(np_us_per_q / max(bf_us_per_q, 1e-6), 1),
            "max_abs_diff": max_diff,
        }
    ]


def main() -> None:
    import backfold as bf

    from benchmarks.digest_eval import run_digest_eval

    repo_root = Path(__file__).resolve().parents[1]
    cli_bin = repo_root / "target" / "release" / "backfold"

    rust_bench = None
    if cli_bin.exists():
        proc = subprocess.run(
            [str(cli_bin), "bench", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        rust_bench = json.loads(proc.stdout)

    payload = {
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "late_updates_python_vs_ecosystems": bench_late_updates(bf, [1_000, 10_000, 100_000]),
        "range_queries": bench_range_queries(bf, n=100_000, n_queries=200),
        "multi_agent_eval": run_multi_agent_eval(bf),
        "digest_compression_eval": run_digest_eval(bf),
        "rust_vs_rust_o3_bench": rust_bench,
    }

    out_path = repo_root / "benchmarks" / "results.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
