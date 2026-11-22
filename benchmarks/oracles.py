"""Independent naive sequential reference oracles and real ecosystem baselines.

These implementations share ZERO code with `backfold` and serve both as:
1. Mathematical ground-truth oracles for parity verification (`max_abs_diff <= 1e-10`).
2. Real head-to-head competitors (`numpy`, `scipy`, `pandas`, `polars`, `river`)
   on out-of-order late arrivals, retractions, and arbitrary time-range queries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl
from river import linear_model, optim, stats
from scipy import linalg as scipy_linalg

BETTING_GRID = np.array([-0.5, -0.25, -0.1, -0.04, 0.04, 0.1, 0.25, 0.5], dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Moments (Mean, Population Variance, Skewness, Excess Kurtosis, Min/Max)
# ---------------------------------------------------------------------------


def oracle_moments_numpy(xs: np.ndarray) -> dict[str, float]:
    n = int(xs.shape[0])
    if n == 0:
        return {"n": 0, "mean": 0.0, "variance": 0.0, "skewness": 0.0, "excess_kurtosis": 0.0}
    mean = float(np.mean(xs))
    d = xs - mean
    m2 = float(np.sum(d * d))
    m3 = float(np.sum(d * d * d))
    m4 = float(np.sum(d * d * d * d))
    var = m2 / n
    skew = (m3 / n) / (var**1.5) if var > 0.0 else 0.0
    kurt = (m4 / n) / (var * var) - 3.0 if var > 0.0 else 0.0
    return {
        "n": n,
        "mean": mean,
        "variance": var,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "min": float(np.min(xs)),
        "max": float(np.max(xs)),
    }


def baseline_moments_pandas(xs: np.ndarray) -> dict[str, float]:
    s = pd.Series(xs)
    return {
        "n": len(s),
        "mean": float(s.mean()),
        "variance": float(s.var(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def baseline_moments_polars(xs: np.ndarray) -> dict[str, float]:
    s = pl.Series("x", xs)
    return {
        "n": int(s.len()),
        "mean": float(s.mean()),
        "variance": float(s.var(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def baseline_moments_river(xs: Sequence[float]) -> dict[str, float]:
    mean = stats.Mean()
    var = stats.Var(ddof=0)
    mn = stats.Min()
    mx = stats.Max()
    for x in xs:
        xf = float(x)
        mean.update(xf)
        var.update(xf)
        mn.update(xf)
        mx.update(xf)
    return {
        "n": len(xs),
        "mean": float(mean.get()),
        "variance": float(var.get()),
        "min": float(mn.get()),
        "max": float(mx.get()),
    }


# ---------------------------------------------------------------------------
# 2. Continuous-Time Irregular-dt EWMA
# ---------------------------------------------------------------------------


def oracle_ewma_numpy(ts: np.ndarray, xs: np.ndarray, decay_rate: float) -> dict[str, float]:
    if ts.shape[0] == 0:
        return {"n": 0, "w_sum": 0.0, "mean": 0.0, "variance": 0.0}
    t_last = ts[-1]
    dt = np.maximum(0.0, (t_last - ts).astype(np.float64))
    arg = decay_rate * dt
    w = np.where(arg > 700.0, 0.0, np.exp(-np.minimum(arg, 700.0)))
    w_sum = float(np.sum(w))
    if w_sum == 0.0:
        return {"n": int(ts.shape[0]), "w_sum": 0.0, "mean": 0.0, "variance": 0.0}
    mean = float(np.sum(w * xs) / w_sum)
    var = float(np.sum(w * (xs - mean) ** 2) / w_sum)
    return {"n": int(ts.shape[0]), "w_sum": w_sum, "mean": mean, "variance": var}


def baseline_ewma_pandas(ts: np.ndarray, xs: np.ndarray, half_life: float) -> float:
    idx = pd.to_datetime(ts, unit="s")
    s = pd.Series(xs, index=idx)
    return float(s.ewm(halflife=f"{half_life}s", times=idx).mean().iloc[-1])


# ---------------------------------------------------------------------------
# 3. Two-Sided Page CUSUM
# ---------------------------------------------------------------------------


def oracle_cusum_sequential(
    ts: np.ndarray,
    xs: np.ndarray,
    mu0: float,
    sigma0: float,
    slack_k: float,
    threshold: float | None = None,
) -> dict[str, float | int | None]:
    s_pos = 0.0
    s_neg = 0.0
    peak_pos = 0.0
    peak_neg = 0.0
    first_alarm_ts: int | None = None
    for t, x in zip(ts.tolist(), xs.tolist()):
        z = (x - mu0) / sigma0
        s_pos = max(0.0, s_pos + (z - slack_k))
        s_neg = max(0.0, s_neg + (-z - slack_k))
        peak_pos = max(peak_pos, s_pos)
        peak_neg = max(peak_neg, s_neg)
        if threshold is not None and first_alarm_ts is None and max(s_pos, s_neg) >= threshold:
            first_alarm_ts = int(t)
    return {
        "current_pos": s_pos,
        "current_neg": s_neg,
        "peak_pos": peak_pos,
        "peak_neg": peak_neg,
        "max_peak": max(peak_pos, peak_neg),
        "first_alarm_ts": first_alarm_ts,
    }


# ---------------------------------------------------------------------------
# 4. Anytime-Valid Mixture E-Process & e-CUSUM Detector
# ---------------------------------------------------------------------------


def oracle_eprocess_sequential(zs: np.ndarray) -> dict[str, float]:
    k = BETTING_GRID.shape[0]
    cum = np.zeros(k, dtype=np.float64)
    edet = np.zeros(k, dtype=np.float64)
    peak_edet = np.zeros(k, dtype=np.float64)
    for z in zs.tolist():
        g = math.tanh(0.5 * z)
        inc = np.log1p(BETTING_GRID * g)
        cum += inc
        edet = np.maximum(0.0, edet + inc)
        peak_edet = np.maximum(peak_edet, edet)

    def logmeanexp(v: np.ndarray) -> float:
        m = float(np.max(v))
        return m + math.log(float(np.mean(np.exp(v - m))))

    log_w = logmeanexp(cum)
    log_ed = logmeanexp(edet)
    peak_valid = math.exp(float(np.max(peak_edet)) - math.log(k))
    return {
        "log_wealth": log_w,
        "wealth": math.exp(min(log_w, 700.0)),
        "log_e_detector": log_ed,
        "e_detector": math.exp(min(log_ed, 700.0)),
        "peak_e_detector_valid": peak_valid,
    }


# ---------------------------------------------------------------------------
# 5. Exact Joseph-Form Sequential Kalman Filter (Constant-Velocity 2D & Generic)
# ---------------------------------------------------------------------------


def oracle_kalman_cv_numpy(
    dts: np.ndarray,
    ys: np.ndarray,
    q_accel: float,
    r_obs: float,
    m0: np.ndarray,
    p0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m = m0.astype(np.float64).copy()
    p = p0.astype(np.float64).copy()
    h = np.array([1.0, 0.0], dtype=np.float64)
    eye = np.eye(2, dtype=np.float64)
    for dt, y in zip(dts.tolist(), ys.tolist()):
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = q_accel * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )
        m_pred = f @ m
        p_pred = f @ p @ f.T + q
        if not math.isnan(y):
            s = float(h @ p_pred @ h + r_obs)
            k = (p_pred @ h) / s
            m = m_pred + k * (y - float(h @ m_pred))
            ikh = eye - np.outer(k, h)
            p = ikh @ p_pred @ ikh.T + np.outer(k, k) * r_obs
            p = 0.5 * (p + p.T)
        else:
            m, p = m_pred, p_pred
    return m, p


# ---------------------------------------------------------------------------
# 6. Continuous-Time Exponentially Discounted Online Ridge Regression
# ---------------------------------------------------------------------------


def oracle_ridge_numpy(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    decay_rate: float,
    alpha: float,
) -> np.ndarray:
    t_last = ts[-1]
    dt = np.maximum(0.0, (t_last - ts).astype(np.float64))
    arg = decay_rate * dt
    w = np.where(arg > 700.0, 0.0, np.exp(-np.minimum(arg, 700.0)))
    xw = xs * w[:, None]
    gram = xw.T @ xs + alpha * np.eye(xs.shape[1], dtype=np.float64)
    rhs = xw.T @ ys
    return np.linalg.solve(gram, rhs)


def baseline_ridge_scipy(
    ts: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    decay_rate: float,
    alpha: float,
) -> np.ndarray:
    t_last = ts[-1]
    dt = np.maximum(0.0, (t_last - ts).astype(np.float64))
    arg = decay_rate * dt
    w = np.where(arg > 700.0, 0.0, np.exp(-np.minimum(arg, 700.0)))
    xw = xs * w[:, None]
    gram = xw.T @ xs + alpha * np.eye(xs.shape[1], dtype=np.float64)
    rhs = xw.T @ ys
    c, lower = scipy_linalg.cho_factor(gram)
    return scipy_linalg.cho_solve((c, lower), rhs)


def baseline_linear_river(xs: np.ndarray, ys: np.ndarray, l2: float = 1e-2) -> list[float]:
    model = linear_model.LinearRegression(
        optimizer=optim.SGD(0.01), l2=l2, intercept_lr=0.0
    )
    d = xs.shape[1]
    keys = [f"f{i}" for i in range(d)]
    for row, y in zip(xs.tolist(), ys.tolist()):
        model.learn_one(dict(zip(keys, row)), float(y))
    return [float(model.weights.get(k, 0.0)) for k in keys]
