# backfold

[![CI](https://github.com/pgg898200-ship-it/backfold/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg898200-ship-it/backfold/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Persistent $O(\log T)$ out-of-order, retractable, forkable online statistical inference, exact RTS Bayesian smoothing & lossless-recoverable AI agent stream compression.**

Pure safe Rust (`#![forbid(unsafe_code)]`) · Zero-copy Python bindings · Sub-5ms native CLI · [Docs & `llms.txt`](https://pgg898200-ship-it.github.io/backfold/)

---

> **Every streaming library assumes time moves forward, nothing gets retracted, and only one consumer reads the stream.**
> Real sensor pipelines, financial feeds, telemetry firehoses, and multi-agent LLM swarms break all three assumptions. When an event arrives 50,000 ticks late—or gets retracted, restated, or branched by 64 counterfactual agents—`river`, `pandas.ewm`, and sequential `numpy`/`scipy` Kalman filters/smoothers fall back to **$O(T)$ full forward-backward replay**.
>
> `backfold` compiles every operator into an **exact associative monoid** inside a persistent copy-on-write B+ fold tree (`FoldTree<M, B>`): **$O(\log T)$ late edits & retractions, $O(\log T)$ exact Rauch–Tung–Striebel (RTS) Bayesian smoothing & marginal log-likelihood $\log p(y_{1:T})$, $O(\log T)$ arbitrary range queries, $O(1)$ zero-copy `fork()` (~16 ns), echo-proof multi-agent CRDT merges, distribution-free e-BH FDR control, and `<300`-token lossless-recoverable LLM stream digests — with zero approximation (`max |diff| <= 1e-12`).**

---

## 30-Second Quickstart

### Python (`pip install backfold`)
```python
import numpy as np
import backfold as bf

ts = np.arange(100_000, dtype=np.int64)
dts = np.ones(100_000)
ys = np.cumsum(np.random.default_rng(0).normal(0.05, 0.1, 100_000))
m0, p0 = [0.0, 0.0], [10.0, 0.0, 0.0, 10.0]

# 1. Exact 2D Constant-Velocity Kalman filter + O(log T) RTS smoother & marginal log-likelihood
kf = bf.KalmanStream(dim=2)
kf.ingest_constant_velocity(ts, ys, dts, q_accel=1e-3, r_obs=0.25)

# 2. O(1) zero-copy counterfactual fork (~16 ns, 0 bytes copied) + O(log T) late repair + RTS smooth (~12.4 µs)
branch = kf.fork()
branch.upsert_constant_velocity(33_333, float(ys[33_333] + 10.0), 1.0, 1e-3, 0.25)
sm_mean, sm_cov = branch.smooth_at(50_000, m0, p0)          # Exact Rauch-Tung-Striebel p(x_t | y_{1:T})
log_ev = branch.log_marginal_likelihood(m0, p0)             # Exact log p(y_{1:T}), O(1) at root!

# 3. Exact 106-bit compensated multivariate covariance, correlation & Mahalanobis anomaly scoring
cov = bf.CovarianceStream(dim=8)
cov.ingest(ts[:10_000], np.random.default_rng(1).normal(1e12, 1.0, size=(10_000, 8)))
d2 = cov.mahalanobis_sq([1e12 + 10.0] * 8)

# 4. Lossless-recoverable <300-token digest for LLM / coding agents
digest = bf.digest_stream("sensor_01", ts, ys, top_k=5, threshold=18.0)
print(digest["markdown"])  # 2,510× smaller than raw CSV, 100% acute-burst window recall
```

### Native CLI (sub-5ms cold startup, zero runtime dependencies)
```sh
cargo build --release --bin backfold

# Run 6 live verified 100,000-tick demos in 0.56 s (0 approximations, 0 downloads)
./target/release/backfold demo

# Compress a 100,000-row CSV (~600k tokens) into a ~239-token causal Markdown digest (2,510× compression)
./target/release/backfold digest benchmarks/sample_100k.csv

# Losslessly drill down to exact raw ticks for any flagged hotspot window
./target/release/backfold drill benchmarks/sample_100k.csv --start 22498 --end 22507

# Reproduce the live Rust O(log T) vs. handwritten -O3 sequential Rust benchmark
./target/release/backfold bench
```

Runnable recipes in [`examples/`](examples/):
- [`examples/kalman_rts_smoothing.py`](examples/kalman_rts_smoothing.py) — $O(\log T)$ RTS smoothing, variance tightening & Bayes evidence model selection
- [`examples/multivariate_anomaly_cov.py`](examples/multivariate_anomaly_cov.py) — `1e12`-offset 8D streaming covariance, regime correlation & batch Mahalanobis scoring
- [`examples/polars_late_arrivals.py`](examples/polars_late_arrivals.py) — retroactive CUSUM alarms, zero-lookahead bitemporal audit, e-BH FDR & agent digest

---

## Verified Live Benchmarks (`[LIVE-CPU]`, 2-vCPU Linux x86-64, `v0.2.0`)

Every number below is measured live on this hardware ([`benchmarks/results.json`](benchmarks/results.json)) and asserts **exact numerical parity** against independent naive sequential oracles ([`benchmarks/oracles.py`](benchmarks/oracles.py)). Python timings conservatively measure a full **edit + query + restore** round-trip (two tree updates + one query).

### 1. Out-of-Order Late Edit Latency — Python API (`T = 100,000`, `max |diff| <= 1e-12`)

| Operator | `backfold` (Python) | NumPy | SciPy | Pandas | Polars | River (`T=10k`) | Speedup | Max `\|diff\|` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **2D Kalman RTS Smooth + $\log p(y)$** | **12.45 µs** | 6,637,232 µs | — | — | — | — | **533,197× vs NumPy RTS** | `1.16e-14` |
| **2D Kalman CV Filter** | **9.32 µs** | 4,737,933 µs | — | — | — | — | **508,471× vs NumPy** | `9.09e-13` |
| **Page CUSUM + 1st alarm** | **5.35 µs** | 134,250 µs *(Py seq)* | — | — | — | — | **25,075× vs seq** | `1.78e-15` |
| **CT Irregular-$\Delta t$ EWMA** | **4.79 µs** | 1,904.5 µs | — | 4,966.7 µs | — | — | **398× NumPy · 1,037× Pandas** | `4.44e-16` |
| **8D Covariance Matrix** | **15.51 µs** | 5,367.8 µs | — | — | — | — | **346× vs NumPy `cov`** | `6.66e-16` |
| **8D Discounted Ridge** | **17.68 µs** | 5,668.3 µs | 6,260.5 µs | — | — | 97,574 µs *(10k)* | **321× NumPy · 354× SciPy · 6,460× River** | `1.33e-15` |
| **Compensated Moments** *(mean/var/skew/kurt)* | **10.51 µs** | 806.0 µs | — | 1,275.8 µs | 263.4 µs | 8,148 µs *(10k)* | **77× NumPy · 121× Pandas · 25× Polars** | `4.57e-16` |

### 2. Rust-vs-Rust Head-to-Head (`backfold` $O(\log T)$ vs. Handwritten `-O3` Sequential Rust)

Not just "Rust vs. Python" — even against tight handwritten `-O3` Rust loops, $O(\log T)$ associative monoids win by **two to four orders of magnitude**:

| Operator | `T = 10,000` | `T = 100,000` | `T = 500,000` | `backfold` @ 500k | Seq Rust `-O3` @ 500k | Max `\|diff\|` |
|---|---:|---:|---:|---:|---:|---:|
| **CT Irregular-$\Delta t$ EWMA** | 124× | 1,016× | **5,020×** | **1.32 µs** | 6,624.1 µs | `2.95e-17` |
| **1D Kalman RW** (`KalmanFilter1D`) | 129× | 1,113× | **4,888×** | **1.22 µs** | 5,967.7 µs | `1.67e-16` |
| **2D Kalman CV** (`KalmanFilter2D`) | 82× | 719× | **3,260×** | **3.34 µs** | 10,898.2 µs | `2.78e-16` |
| **2D Kalman RTS Smooth + $\log p(y)$** | 79× | 639× | **3,109×** | **6.52 µs** | 20,262.2 µs | `3.06e-14` |
| **Page CUSUM + 1st alarm** | 32× | 276× | **1,372×** | **1.89 µs** | 2,587.6 µs | `1.78e-15` |
| **8D Discounted Ridge** | 79× | **640×** | — | *6.22 µs @ 100k* | *3,982.3 µs @ 100k* | `2.00e-15` |
| **8D Compensated Covariance** | 26× | **234×** | — | *5.69 µs @ 100k* | *1,330.0 µs @ 100k* | `1.93e-14` |
| **Compensated Moments** | 11× | 88× | **414×** | **3.77 µs** | 1,564.2 µs | `3.12e-14` |
| **Arbitrary 60% Range Query** | — | 65× | **344×** | **2.76 µs** | 948.8 µs | `9.21e-15` |

### 3. Multi-Agent Coordination, Echo Immunity & 500-Stream FDR Control

| Scenario | Naive Baseline | `backfold` | Edge |
|---|---|---|---|
| **16-Agent Gossip Swarm** *(2,000 events echoed 3× out-of-order)* | 66.7% overwrite churn (4,000 overwrites); `1.732×` ($\sqrt{3}$) echo z-inflation | **0.0% overwrite churn**; 4,000/4,000 echoes deduplicated; **1.000× exact evidence** | Echo-proof `(event_time, source_id, seq)` CRDT |
| **500-Stream Firehose** *(10 true shifts + 50 3×-loud traps, $\alpha=0.05$)* | 63 alarms: 10 TP, **53 FP (50/50 loud traps!)**, **FDR = 84.1%** | **10 alarms: 10/10 TP, 0 FP, 0/50 loud traps, FDR = 0.0%** (`259 ms` total) | Causal MAD + composite-null Shiryaev-Roberts e-value + Wang-Ramdas e-BH |
| **64-Agent Counterfactual Forking** *(`T=100k`, 5 edits/agent)* | 3,937.5 ms (`61.5 ms/agent`) | **1.27 ms** (`0.0198 ms/agent`, `max_diff = 3.55e-15`) | **3,110× faster** ($O(1)$ persistent `Arc` fork) |

### 4. Lossless-Recoverable AI Agent Stream Compression (`50 streams × 20,000 ticks`)

| Representation | Avg Tokens | Compression | Acute Burst Exact-Window Recall | Regime Shift Recall | Loud 3×-Distractor FP Rate | Lossless Recovery |
|---|---:|---:|---:|---:|---:|---|
| Raw CSV | 120,000 | 1× | — *(overflows context)* | — | — | Yes |
| `pandas.describe()` | 17 | 7,059× | **0.0%** *(drops timestamps)* | 100.0% | **100.0%** | No |
| Uniform 200× Downsample | 600 | 200× | **20.0%** *(misses 5-tick spikes)* | 86.7% | **30.0%** | No |
| **`backfold digest`** | **255.8** | **469.1×** *(2,510× on 100k)* | **100.0%** | **100.0%** | **0.0%** | **Yes (`backfold drill`)** |

---

## What's Inside (7 Exact Associative Operators + Swarm CRDT)

1. **`KalmanStream` / `KalmanFilter1D` / `KalmanFilter2D` / `KalmanMonoid`** — Exact Särkkä & García-Fernández (IEEE TAC 2020) associative Bayesian filter **and exact $O(\log T)$ Rauch–Tung–Striebel (RTS) smoother (`smooth_at`, `smooth_many`) + exact marginal log-likelihood (`log_marginal_likelihood`, `range_log_marginal_likelihood`)**. Hardened with **Anchor-Centered Associative Gaussian Potentials** (eliminating $O(y^2/S)$ cancellation down to `1.14e-15` relative error on 100k drifting streams and `1.04e-12` at `1e6` offset), Joseph-form PSD leaves, zero-alloc stack `1D`/`2D` fast paths, and x86-64 subnormal flush (`|x| < 1e-290 => 0`).
2. **`CovarianceStream` / `CovarianceMonoid`** — Exact multivariate streaming covariance, Pearson correlation, arbitrary `[t1, t2]` regime correlation, and Jacobi-equilibrated Cholesky **Mahalanobis anomaly scoring** (`mahalanobis_sq`, factored batch `mahalanobis_many`), using 106-bit Knuth TwoSum compensated mean vectors and zero-alloc inline stack storage for `dim <= 8`.
3. **`MomentsStream` / `MomentsMonoid`** — Centered Chan–Pébay–Terriberry higher-order moments (`mean`, `variance`, `skewness`, `excess_kurtosis`, `min`/`max` + exact timestamps) with **Knuth TwoSum 106-bit mean compensation** (`<1e-12` variance accuracy even on `1e12 + N(0, 1)` streams).
4. **`EwmaStream` / `EwmaMonoid`** — Continuous-time irregular-$\Delta t$ centered exponentially weighted mean & variance with subnormal-safe decay clamping (`lambda * dt > 700 => 0`).
5. **`CusumStream` / `CusumMonoid`** — Two-sided tropical-semiring Page CUSUM with non-empty `max_pref` (identity = $-\infty$) supporting exact **$O(\log T)$ earliest-alarm binary descent** (`first_alarm(threshold)`).
6. **`EProcessStream` / `EProcessMonoid` + `ebh_reject`** — Simultaneously tracks Waudby-Smith–Ramdas 8-point symmetric betting martingale (`wealth`), Shin–Ramdas–Rinaldo Page e-CUSUM (`e_detector`), and exact log-sum-exp **Shiryaev–Roberts (`sr_evalue`) + all-subinterval scan (`scan_evalue`)** yielding a provably valid composite-null e-value ($\mathbb{E}_0[e] \le 1$, even under heavy-tailed Cauchy/Student-$t$ noise) for Wang–Ramdas arbitrary-dependence **e-BH FDR control**.
7. **`RidgeStream` / `RidgeMonoid`** — Continuous-time exponentially discounted Ridge / RLS regression with zero-alloc inline stack storage for `dim <= 8`, in-place `combine_assign`, **Jacobi-equilibrated Cholesky** ($D^{-1/2}(G + \alpha I)D^{-1/2}$) with pivoted LU & relative-Tikhonov fallback, and exact predictive variance.
8. **`SwarmEProcessLedger` & `BitemporalMoments`** — Echo-proof multi-agent CRDT ledger (`Arc::ptr_eq`-accelerated idempotent merge) and zero-lookahead bitemporal `as_of(knowledge_time)` snapshots.

---

## Honest Crossovers & Limitations

`backfold` is **not** magic for every workload. Know where it wins and where a simpler tool wins:

- **Pure append-only, single-pass, in-order aggregation:** If you never receive late arrivals, never retract/edit history, never query arbitrary sub-intervals `[t1, t2]`, and never fork branches, a plain sequential loop is faster on initial ingest. On `MomentsMonoid` at `T = 100k–500k`, bulk tree construction takes `10.2 ms` (`T=100k`) / `47.9 ms` (`T=500k`) vs. `0.30 ms` / `1.56 ms` for one sequential Rust pass — breaking even after **31–34 late updates or range queries**.
- **High-dimensional dense matrices (`d > 64`):** `RidgeMonoid` and `CovarianceMonoid` maintain packed upper-triangular $O(d^2)$ matrices per tree node. They shine for streaming telemetry, control, sensor fusion, and factor models (`d <= 32`, zero-alloc inline for `d <= 8`), not 10,000-feature dense NLP embeddings.
- **Exact quantile / median queries:** Order statistics (arbitrary quantiles) do not admit a fixed-size exact associative monoid without sketch approximation. `backfold` sticks strictly to **exact** operators (`causal_robust_scale` computes exact median/MAD on causal warm-up prefixes).

---

## For AI Coding Agents

- **Agent discovery index:** [`llms.txt`](llms.txt) and complete API reference [`llms-full.txt`](llms-full.txt) (pure Markdown).
- **Drop-in coding-agent skill:** [`skills/backfold/SKILL.md`](skills/backfold/SKILL.md) (works with Claude Code, Cursor, Codex, OpenCode, and MCP-compatible agents).
- **Technical deep-dive:** [`docs/SUBNORMAL_KALMAN.md`](docs/SUBNORMAL_KALMAN.md) (x86-64 subnormal traps, Anchor-Centered Gaussian Potentials for $O(\log T)$ RTS smoothing, 106-bit TwoSum moments, and Shiryaev–Roberts e-values).
- **Reproduce all tests & benchmarks:**
  ```sh
  cargo test --package backfold-core --package backfold-cli --release
  cargo clippy --workspace --all-targets -- -D warnings
  pytest -v tests/
  python -m benchmarks.run_benchmarks
  ```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
