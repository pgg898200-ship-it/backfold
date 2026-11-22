---
name: backfold
description: Persistent O(log T) out-of-order, retractable, forkable online statistical inference (Kalman filter + exact RTS smoother + marginal log-likelihood, multivariate Covariance/Mahalanobis, CUSUM, EWMA, Ridge, anytime e-values + e-BH FDR) and lossless-recoverable <300-token stream compression for AI agents. Use whenever inspecting large CSV/time-series telemetry streams, repairing late-arriving or retracted stream events, running RTS smoothing or multi-agent counterfactual simulations, or preventing multi-stream false-alarm / echo-inflation failures.
---

# backfold — Agent Streaming & Telemetry Skill

Use `backfold` (`pip install backfold`) instead of dumping raw CSV / time-series rows into your context window or re-scanning history in Python/NumPy/Pandas loops.

## Rule 1: Never read raw large telemetry CSVs directly into context
When given a time-series or telemetry CSV (`>500` rows), run `backfold digest` first:
```sh
backfold digest <path/to/stream.csv> --value-col 1 --top-k 5
```
This compresses `20,000–100,000` ticks into a **~240–260 token** pure-Markdown causal digest (**469×–2,510× compression**):
- Preserves **100% exact-window recall** on rare 5-tick acute bursts and sustained regime shifts.
- Uses causal warm-up `median + MAD` scale + composite-null Shiryaev-Roberts e-values so **3×-louder distractor channels cause 0% false alarms**.
- Every flagged hotspot includes exact `[t_start..t_end]` cursors. Recover exact raw rows losslessly with:
```sh
backfold drill <path/to/stream.csv> --start <t_start> --end <t_end>
```

## Rule 2: Use `backfold` Python bindings for out-of-order, retractable, or forkable streaming inference
When writing Python code involving late arrivals, retractions, delayed labels, exact RTS smoothing, online Bayesian model selection, arbitrary `[t1, t2]` historical range queries, or multi-agent counterfactual branches:

| Task | `backfold` Class | Verified Latency (`T=100k` late edit) | Exact Speedup |
|---|---|---|---|
| Exact RTS Kalman smoother + `log p(y)` | `bf.KalmanStream(dim=2).smooth_at(...)` / `log_marginal_likelihood(...)` | **12.45 µs** Py (`5.78 µs` Rust) | **533,197×** vs NumPy RTS (`3,109×` vs `-O3` Rust @500k) |
| 1D / 2D / ND Kalman filter (irregular `dt`, missing obs) | `bf.KalmanStream(dim=2)` | **9.32 µs** Py (`3.03 µs` Rust) | **508,471×** vs NumPy seq (`3,260×` vs `-O3` Rust @500k) |
| Multivariate covariance, correlation & Mahalanobis anomaly (`1e12`-safe) | `bf.CovarianceStream(dim=8)` | **15.51 µs** Py (`5.69 µs` Rust) | **346×** vs NumPy `cov` (`234×` vs `-O3` Rust) |
| Two-sided Page CUSUM + exact earliest alarm | `bf.CusumStream(mu0, sigma0, slack_k)` | **5.35 µs** Py (`1.87 µs` Rust) | **25,075×** vs Py seq (`1,372×` vs `-O3` Rust @500k) |
| Continuous-time irregular-`dt` EWMA | `bf.EwmaStream(half_life=...)` | **4.79 µs** Py (`1.24 µs` Rust) | **398×** NumPy · **1,037×** Pandas (`5,020×` Rust @500k) |
| Continuous-time discounted Ridge / RLS (delayed labels) | `bf.RidgeStream(dim=8, decay_rate=...)` | **17.68 µs** Py (`6.22 µs` Rust) | **321×** NumPy · **354×** SciPy · **6,460×** River@10k |
| Higher-order moments (`1e12` offset-safe) | `bf.MomentsStream()` | **10.51 µs** Py (`3.43 µs` Rust) | **77×** NumPy · **121×** Pandas · **25×** Polars |
| Echo-proof multi-agent evidence fusion | `bf.SwarmEProcessLedger()` | 0% overwrite churn, 0% echo inflation | Eliminates blackboard overwrites & gossip double-counting |
| Multi-stream FDR control (`alpha = 0.05`) | `bf.EProcessStream()` + `bf.ebh_reject(es, 0.05)` | **FDR = 0.000, Recall = 1.000** on 500 channels | Arbitrary-dependence Wang-Ramdas e-BH guarantee |
| Zero-lookahead point-in-time backtesting | `bf.BitemporalMoments()` | $O(\log V)$ `as_of(knowledge_time)` | 0% lookahead leakage |

## Rule 3: Know when NOT to use `backfold`
- Pure append-only, single-pass, in-order aggregation with zero retractions, zero range queries, and zero forks: use a single sequential pass (break-even is ~31–34 late updates or range queries at `T = 100k–500k`).
- High-dimensional Ridge or Covariance (`d > 64`): both store $O(d^2)$ packed matrices per node (optimized for `d <= 32`, zero-alloc inline for `d <= 8`).

## Reference
See `llms-full.txt` in the `backfold` repository for the complete copy-pasteable Python, Rust, and CLI API.
