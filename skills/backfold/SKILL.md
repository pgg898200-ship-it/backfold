---
name: backfold
description: Persistent O(log T) out-of-order, retractable, forkable online statistical inference (Kalman, CUSUM, EWMA, Ridge, anytime e-values + e-BH FDR) and lossless-recoverable <300-token stream compression for AI agents. Use whenever inspecting large CSV/time-series telemetry streams, repairing late-arriving or retracted stream events, running multi-agent counterfactual simulations, or preventing multi-stream false-alarm / echo-inflation failures.
---

# backfold — Agent Streaming & Telemetry Skill

Use `backfold` instead of dumping raw CSV / time-series rows into your context window or re-scanning history in Python/NumPy/Pandas loops.

## Rule 1: Never read raw large telemetry CSVs directly into context
When given a time-series or telemetry CSV (`>500` rows), run `backfold digest` first:
```sh
backfold digest <path/to/stream.csv> --col 1 --top-k 5
```
This compresses `20,000–100,000` ticks into a **~240–260 token** pure-Markdown causal digest (**469×–2,510× compression**):
- Preserves **100% exact-window recall** on rare 5-tick acute bursts and sustained regime shifts.
- Uses causal warm-up `median + MAD` scale + composite-null Shiryaev-Roberts e-values so **3×-louder distractor channels cause 0% false alarms**.
- Every flagged hotspot includes exact `[t_start..t_end]` cursors. Recover exact raw rows losslessly with:
```sh
backfold drill <path/to/stream.csv> --start <t_start> --end <t_end>
```

## Rule 2: Use `backfold` Python bindings for out-of-order, retractable, or forkable streaming inference
When writing Python code involving late arrivals, retractions, delayed labels, arbitrary `[t1, t2]` historical range queries, or multi-agent counterfactual branches:

| Task | `backfold` Class | Verified Latency (`T=100k` late edit) | Exact Speedup |
|---|---|---|---|
| 1D / 2D / ND Kalman filter (irregular `dt`, missing obs) | `bf.KalmanStream(dim=2)` | **8.72 µs** Py (`2.67 µs` Rust) | **512,577×** vs NumPy seq (`847×` vs `-O3` Rust) |
| Two-sided Page CUSUM + exact earliest alarm | `bf.CusumStream(mu0, sigma0, slack_k)` | **5.32 µs** Py (`1.88 µs` Rust) | **15,378×** vs Py seq (`269×` vs `-O3` Rust) |
| Continuous-time irregular-`dt` EWMA | `bf.EwmaStream(half_life=...)` | **5.03 µs** Py (`1.19 µs` Rust) | **413×** NumPy · **1,034×** Pandas |
| Continuous-time discounted Ridge / RLS (delayed labels) | `bf.RidgeStream(dim=8, decay_rate=...)` | **17.47 µs** Py (`6.08 µs` Rust) | **533×** NumPy · **567×** SciPy · **6,618×** River@10k |
| Higher-order moments (`1e12` offset-safe) | `bf.MomentsStream()` | **10.63 µs** Py (`3.41 µs` Rust) | **91×** NumPy · **146×** Pandas · **28×** Polars |
| Echo-proof multi-agent evidence fusion | `bf.SwarmEProcessLedger()` | 0% overwrite churn, 0% echo inflation | Eliminates blackboard overwrites & gossip double-counting |
| Multi-stream FDR control (`alpha = 0.05`) | `bf.EProcessStream()` + `bf.ebh_reject(es, 0.05)` | **FDR = 0.000, Recall = 1.000** on 500 channels | Arbitrary-dependence Wang-Ramdas e-BH guarantee |
| Zero-lookahead point-in-time backtesting | `bf.BitemporalMoments()` | $O(\log V)$ `as_of(knowledge_time)` | 0% lookahead leakage |

## Rule 3: Know when NOT to use `backfold`
- Pure append-only, single-pass, in-order aggregation with zero retractions, zero range queries, and zero forks: use a single sequential pass (break-even is ~34–39 late updates or range queries at `T = 100k–500k`).
- High-dimensional Ridge (`d > 64`): `RidgeMonoid` stores $O(d^2)$ packed Gram matrices per node (optimized for `d <= 32`, zero-alloc inline for `d <= 8`).

## Reference
See `llms-full.txt` in the `backfold` repository for the complete copy-pasteable Python, Rust, and CLI API.
