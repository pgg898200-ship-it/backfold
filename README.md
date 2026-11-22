# backfold

[![CI](https://github.com/pgg898200-ship-it/backfold/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg898200-ship-it/backfold/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/backfold.svg)](https://pypi.org/project/backfold/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Persistent $O(\log T)$ out-of-order, retractable, forkable online statistical inference, exact RTS Bayesian smoothing, streaming PCA, native MCP server & lossless-recoverable AI agent stream compression.**

Pure safe Rust (`#![forbid(unsafe_code)]`) · Zero-copy GIL-free Python · Sub-5ms native CLI & MCP server · **[Live WebAssembly Playground & `llms.txt`](https://pgg898200-ship-it.github.io/backfold/)**

---

> **Every streaming library assumes time moves forward, nothing gets retracted, and only one consumer reads the stream.**
> Real sensor pipelines, financial feeds, telemetry firehoses, and multi-agent LLM swarms break all three assumptions. When an event arrives 50,000 ticks late—or gets retracted, restated, or branched by 64 counterfactual agents—`river`, `pandas.ewm`, and sequential `numpy`/`scipy` Kalman filters/smoothers fall back to **$O(T)$ full forward-backward replay**.
>
> `backfold` compiles every operator into an **exact associative monoid** inside a persistent copy-on-write B+ fold tree (`FoldTree<M, B>`): **$O(\log T)$ late edits & retractions, $O(\log T)$ exact Rauch–Tung–Striebel (RTS) Bayesian smoothing & marginal log-likelihood $\log p(y_{1:T})$, zero-blind-spot Bayesian changepoint shock localization, 106-bit streaming PCA & Mahalanobis anomaly scoring, $O(1)$ zero-copy `fork()` (~20 ns), echo-proof multi-agent CRDT merges, distribution-free e-BH FDR control (`sift_streams`), a zero-dependency stdio MCP server (`backfold mcp`), and `<300`-token lossless-recoverable LLM stream digests — with zero approximation (`max |diff| <= 1e-12`).**

---

## Live Interactive WebAssembly Playground

Try real compiled Rust `backfold-core` (`backfold_wasm.wasm`, zero backend server) directly in your browser on 20,000 ticks — click to inject late glitches, Shift+Click to retract ticks, localize off-grid shocks, and benchmark **WASM $O(\log T)$ (`~8 µs`) vs. JavaScript sequential RTS (`~28 ms`, ~3,600× faster, `diff < 1e-14`)** live on your own machine:

**[https://pgg898200-ship-it.github.io/backfold/](https://pgg898200-ship-it.github.io/backfold/)**

---

## 30-Second Quickstart

### Python (`pip install backfold` — prebuilt wheels for Linux, macOS, Windows, Py 3.10–3.13)
```python
import numpy as np
import backfold as bf

ts = np.arange(100_000, dtype=np.int64)
dts = np.ones(100_000)
ys = np.cumsum(np.random.default_rng(0).normal(0.05, 0.1, 100_000))
m0, p0 = [0.0, 0.0], [10.0, 0.0, 0.0, 10.0]

# 1. Exact 2D Constant-Velocity Kalman filter + O(log T) RTS smoother + Bayesian changepoint locator
kf = bf.KalmanStream(dim=2)
kf.ingest_constant_velocity(ts, ys, dts, q_accel=1e-3, r_obs=0.25)

# 2. O(1) zero-copy counterfactual fork (~20 ns) + O(log T) late repair + RTS smooth (~11.1 µs)
branch = kf.fork()
branch.upsert_constant_velocity(33_333, float(ys[33_333] + 10.0), 1.0, 1e-3, 0.25)
sm_mean, sm_cov = branch.smooth_at(50_000, m0, p0)                 # Exact RTS p(x_t | y_{1:T})
log_ev = branch.log_marginal_likelihood(m0, p0)                    # Exact log p(y_{1:T}), O(1) at root!
cp = branch.locate_changepoint(m0, p0, [200.0, 0.0, 0.0, 1.0])     # Zero-blind-spot shock locator (<0.6 ms!)

# 3. Exact 106-bit compensated multivariate covariance, cyclic-Jacobi streaming PCA & Mahalanobis
cov = bf.CovarianceStream(dim=8)
cov.ingest(ts[:10_000], np.random.default_rng(1).normal(1e12, 1.0, size=(10_000, 8)))
pca = cov.range_pca(1_000, 9_000, ddof=1, top_k=3)                 # Zero BLAS, exact even at 1e12 offset!
d2 = cov.mahalanobis_sq([1e12 + 10.0] * 8)

# 4. Single-call GIL-free 500-channel Wang-Ramdas e-BH FDR sifter (provable FDR <= alpha)
sift = bf.sift_streams(np.random.default_rng(2).normal(size=(500, 400)), warmup=120, alpha=0.05)

# 5. Lossless-recoverable <300-token digest for LLM / coding agents
digest = bf.digest_stream("sensor_01", ts, ys, top_k=5, threshold=18.0)
print(digest["markdown"])  # 2,510× smaller than raw CSV, 100% acute-burst window recall
```

### Zero-Dependency Native MCP Server (`backfold mcp`)
Drop the precompiled sub-1 MB `backfold` binary straight into **Claude Desktop**, **Cursor**, **Windsurf**, or **Claude Code** (zero Python, Node, or Docker runtime required):
```json
{
  "mcpServers": {
    "backfold": {
      "command": "backfold",
      "args": ["mcp"]
    }
  }
}
```
Exposes 5 production streaming tools over newline-delimited stdio JSON-RPC 2.0 (`MCP 2024-11-05`):
- `backfold_digest_csv` — `<300`-token lossless-recoverable Markdown statistical digest
- `backfold_drill_csv` — exact 106-bit range moments + raw ticks for `[start, end]`
- `backfold_kalman_rts_csv` — $O(\log T)$ RTS smoothing + $\log p(y_{1:T})$ + zero-blind-spot shock localization
- `backfold_covariance_pca_csv` — 106-bit streaming covariance, correlation, cyclic-Jacobi PCA & top Mahalanobis outliers
- `backfold_ebh_sift_csv` — Wang–Ramdas arbitrary-dependence e-BH FDR-controlled multi-channel alarm sifter

### Native CLI (sub-5ms cold startup, zero runtime dependencies)
```sh
cargo build --release --bin backfold

# Run 7 live verified 100,000-tick demos in 0.53 s (0 approximations, 0 downloads)
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
- [`examples/streaming_pca_and_changepoint_bf.py`](examples/streaming_pca_and_changepoint_bf.py) — zero-blind-spot Bayesian shock localization (`+1e6` offset in `0.39 ms`), `1e12` range PCA & 500-channel `sift_streams`
- [`examples/multivariate_anomaly_cov.py`](examples/multivariate_anomaly_cov.py) — `1e12`-offset 8D streaming covariance, regime correlation & batch Mahalanobis scoring
- [`examples/polars_late_arrivals.py`](examples/polars_late_arrivals.py) — retroactive CUSUM alarms, zero-lookahead bitemporal audit, e-BH FDR & agent digest

---

## Verified Live Benchmarks (`[LIVE-CPU]`, 2-vCPU Linux x86-64, `v0.3.0`)

Every number below is measured live on this hardware ([`benchmarks/results.json`](benchmarks/results.json)) and asserts **exact numerical parity** against independent naive sequential oracles ([`benchmarks/oracles.py`](benchmarks/oracles.py)). Python timings conservatively measure a full **edit + query + restore** round-trip (two tree updates + one query).

### 1. Out-of-Order Late Edit Latency — Python API (`T = 100,000`, `max |diff| <= 1e-12`)

| Operator | `backfold` (Python) | NumPy | SciPy | Pandas | Polars | River (`T=10k`) | Speedup | Max `\|diff\|` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **2D Kalman RTS Smooth + $\log p(y)$** | **11.12 µs** | 6,437,010 µs | — | — | — | — | **578,868× vs NumPy RTS** | `1.47e-14` |
| **2D Kalman CV Filter** | **8.82 µs** | 4,563,614 µs | — | — | — | — | **517,710× vs NumPy** | `9.09e-13` |
| **Page CUSUM + 1st alarm** | **3.83 µs** | 137,450 µs *(Py seq)* | — | — | — | — | **35,897× vs seq** | `1.78e-15` |
| **CT Irregular-$\Delta t$ EWMA** | **3.92 µs** | 1,940.7 µs | — | 4,831.8 µs | — | — | **495× NumPy · 1,231× Pandas** | `1.78e-15` |
| **8D Covariance Matrix** | **15.60 µs** | 5,589.1 µs | — | — | — | — | **358× vs NumPy `cov`** | `6.66e-16` |
| **8D Discounted Ridge** | **16.95 µs** | 5,567.1 µs | 5,920.6 µs | — | — | 103,981 µs *(10k)* | **328× NumPy · 349× SciPy · 7,092× River** | `1.33e-15` |
| **Compensated Moments** *(mean/var/skew/kurt)* | **6.06 µs** | 730.4 µs | — | 1,293.5 µs | 273.0 µs | 8,215 µs *(10k)* | **121× NumPy · 213× Pandas · 45× Polars** | `4.54e-16` |

### 2. Rust-vs-Rust Head-to-Head (`backfold` $O(\log T)$ vs. Handwritten `-O3` Sequential Rust)

Not just "Rust vs. Python" — even against tight handwritten `-O3` Rust loops, Pareto-tuned (`B = 4`, single-pass `split_summary`, in-place `combine_assign`) associative monoids win by **up to 8,152×** (sub-microsecond at half a million ticks!):

| Operator | `T = 10,000` | `T = 100,000` | `T = 500,000` | `backfold` @ 500k | Seq Rust `-O3` @ 500k | Max `\|diff\|` |
|---|---:|---:|---:|---:|---:|---:|
| **CT Irregular-$\Delta t$ EWMA** | 207× | 1,736× | **8,152×** | **0.806 µs** | 6,570.7 µs | `2.26e-17` |
| **1D Kalman RW** (`KalmanFilter1D`) | 160× | 1,338× | **6,185×** | **0.979 µs** | 6,052.4 µs | `1.39e-16` |
| **2D Kalman RTS Smooth + $\log p(y)$** | 89× | 760× | **3,694×** | **5.367 µs** | 19,823.4 µs | `3.06e-14` |
| **2D Kalman CV** (`KalmanFilter2D`) | 89× | 760× | **3,538×** | **3.110 µs** | 11,000.2 µs | `2.78e-16` |
| **Page CUSUM + 1st alarm** | 53× | 454× | **2,177×** | **1.181 µs** | 2,571.3 µs | `1.78e-15` |
| **Compensated Moments** | 32× | 259× | **1,288×** | **1.278 µs** | 1,646.3 µs | `3.12e-14` |
| **Arbitrary 60% Range Query** | — | 154× | **664×** | **1.404 µs** | 932.2 µs | `8.77e-15` |
| **8D Discounted Ridge** | 79× | **659×** | — | *6.108 µs @ 100k* | *4,022.8 µs @ 100k* | `2.00e-15` |
| **8D Compensated Covariance** | 26× | **262×** | — | *5.677 µs @ 100k* | *1,484.7 µs @ 100k* | `1.93e-14` |

### 3. Multi-Agent Coordination, Echo Immunity & 500-Stream FDR Control

| Scenario | Naive Baseline | `backfold` | Edge |
|---|---|---|---|
| **16-Agent Gossip Swarm** *(2,000 events echoed 3× out-of-order)* | 66.7% overwrite churn (4,000 overwrites); `1.732×` ($\sqrt{3}$) echo z-inflation | **0.0% overwrite churn**; 4,000/4,000 echoes deduplicated; **1.000× exact evidence** | Echo-proof `(event_time, source_id, seq)` CRDT |
| **500-Stream Firehose** *(10 true shifts + 50 3×-loud traps, $\alpha=0.05$)* | 63 alarms: 10 TP, **53 FP (50/50 loud traps!)**, **FDR = 84.1%** | **10 alarms: 10/10 TP, 0 FP, 0/50 loud traps, FDR = 0.0%** (`107.4 ms` via `sift_streams`) | Pure-Rust GIL-free causal MAD + Shiryaev–Roberts e-value + Wang–Ramdas e-BH |
| **64-Agent Counterfactual Forking** *(`T=100k`, 5 edits/agent)* | 4,688.1 ms (`73.3 ms/agent`) | **1.41 ms** (`0.022 ms/agent`, `max_diff = 3.55e-15`) | **3,326× faster** ($O(1)$ persistent `Arc` fork) |

### 4. Lossless-Recoverable AI Agent Stream Compression (`50 streams × 20,000 ticks`)

| Representation | Avg Tokens | Compression | Acute Burst Exact-Window Recall | Regime Shift Recall | Loud 3×-Distractor FP Rate | Lossless Recovery |
|---|---:|---:|---:|---:|---:|---|
| Raw CSV | 120,000 | 1× | — *(overflows context)* | — | — | Yes |
| `pandas.describe()` | 17 | 7,059× | **0.0%** *(drops timestamps)* | 100.0% | **100.0%** | No |
| Uniform 200× Downsample | 600 | 200× | **20.0%** *(misses 5-tick spikes)* | 86.7% | **30.0%** | No |
| **`backfold digest`** | **255.8** | **469.1×** *(2,510× on 100k)* | **100.0%** | **100.0%** | **0.0%** | **Yes (`backfold drill`)** |

---

## What's Inside (7 Exact Associative Operators + Swarm CRDT + Native MCP + WASM)

1. **`KalmanStream` / `KalmanFilter1D` / `KalmanFilter2D` / `KalmanMonoid`** — Exact Särkkä & García-Fernández (IEEE TAC 2020) associative Bayesian filter, **single-pass $O(\log T)$ Rauch–Tung–Striebel (RTS) smoother (`smooth_at`, `smooth_many`)**, exact marginal log-likelihood (`log_marginal_likelihood`, `range_log_marginal_likelihood`), **zero-cancellation Bayesian changepoint Bayes factors (`changepoint_bayes_factor`, `scan_changepoints`)**, and **zero-blind-spot cached-subtree Bayesian shock localization (`locate_changepoint`)**. Hardened with **Anchor-Centered Associative Gaussian Potentials** (`1.14e-15` rel err @ 100k; `1.04e-12` at `1e6` offset), Joseph-form PSD leaves, zero-alloc stack `1D`/`2D` paths, and x86-64 subnormal flush (`|x| < 1e-290 => 0`).
2. **`CovarianceStream` / `CovarianceMonoid`** — Exact multivariate streaming covariance, Pearson correlation, arbitrary `[t1, t2]` regime correlation, **exact pure-safe cyclic-Jacobi streaming PCA (`pca`, `range_pca` — zero BLAS/LAPACK, `<1e-11` accuracy even at `1e12` offset!)**, and Jacobi-equilibrated Cholesky **Mahalanobis anomaly scoring** (`mahalanobis_sq`, factored batch `mahalanobis_many`), with 106-bit Knuth TwoSum compensated mean vectors and zero-alloc inline stack storage for `dim <= 8`.
3. **`MomentsStream` / `MomentsMonoid`** — Centered Chan–Pébay–Terriberry higher-order moments (`mean`, `variance`, `skewness`, `excess_kurtosis`, `min`/`max` + exact timestamps) with **Knuth TwoSum 106-bit mean compensation** (`<1e-12` variance accuracy even on `1e12 + N(0, 1)` streams) and singleton-leaf fast-path `combine_assign`.
4. **`EwmaStream` / `EwmaMonoid`** — Continuous-time irregular-$\Delta t$ centered exponentially weighted mean & variance with subnormal-safe decay clamping (`lambda * dt > 700 => 0`).
5. **`CusumStream` / `CusumMonoid`** — Two-sided tropical-semiring Page CUSUM with non-empty `max_pref` (identity = $-\infty$) supporting exact **$O(\log T)$ earliest-alarm binary descent** (`first_alarm(threshold)`).
6. **`EProcessStream` / `EProcessMonoid` + `ebh_reject` + `sift_streams`** — Simultaneously tracks Waudby-Smith–Ramdas 8-point symmetric betting martingale (`wealth`), Shin–Ramdas–Rinaldo Page e-CUSUM (`e_detector`), and exact log-sum-exp **Shiryaev–Roberts (`sr_evalue`) + all-subinterval scan (`scan_evalue`)** yielding a provably valid composite-null e-value ($\mathbb{E}_0[e] \le 1$, even under heavy-tailed Cauchy/Student-$t$ noise) for Wang–Ramdas arbitrary-dependence **e-BH FDR control**, plus single-call GIL-free batch `sift_streams`.
7. **`RidgeStream` / `RidgeMonoid`** — Continuous-time exponentially discounted Ridge / RLS regression with zero-alloc inline stack storage for `dim <= 8`, in-place `combine_assign`, **Jacobi-equilibrated Cholesky** ($D^{-1/2}(G + \alpha I)D^{-1/2}$) with pivoted LU & relative-Tikhonov fallback, and exact predictive variance.
8. **`SwarmEProcessLedger` & `BitemporalMoments`** — Echo-proof multi-agent CRDT ledger (`Arc::ptr_eq`-accelerated idempotent merge) and zero-lookahead bitemporal `as_of(knowledge_time)` snapshots.
9. **GIL-free Python scaling** — Bulk `ingest*`, `smooth_many`, `scan_changepoints`, `locate_changepoint`, `mahalanobis_many`, `digest_stream`, and `sift_streams` release the Python GIL (`py.allow_threads`) for true multi-core scaling in Python agent swarms.

---

## Honest Crossovers & Limitations

`backfold` is **not** magic for every workload. Know where it wins and where a simpler tool wins:

- **Pure append-only, single-pass, in-order aggregation:** If you never receive late arrivals, never retract/edit history, never query arbitrary sub-intervals `[t1, t2]`, and never fork branches, a plain sequential loop is faster on initial ingest. On `MomentsMonoid` at `T = 100k–500k`, bulk tree construction takes `9.99 ms` (`T=100k`) / `51.07 ms` (`T=500k`) vs. `0.30 ms` / `1.65 ms` for one sequential Rust pass — breaking even after **31–33 late updates or range queries**.
- **High-dimensional dense matrices (`d > 64`):** `RidgeMonoid` and `CovarianceMonoid` maintain packed upper-triangular $O(d^2)$ matrices per tree node. They shine for streaming telemetry, control, sensor fusion, and factor models (`d <= 32`, zero-alloc inline for `d <= 8`), not 10,000-feature dense NLP embeddings.
- **Exact quantile / median queries:** Order statistics (arbitrary quantiles) do not admit a fixed-size exact associative monoid without sketch approximation. `backfold` sticks strictly to **exact** operators (`causal_robust_scale` computes exact median/MAD on causal warm-up prefixes).

---

## For AI Coding Agents

- **Native stdio MCP server:** `backfold mcp` (5 built-in CSV streaming tools, zero runtime dependencies).
- **Agent discovery index:** [`llms.txt`](llms.txt) and complete API reference [`llms-full.txt`](llms-full.txt) (pure Markdown, also hosted at `https://pgg898200-ship-it.github.io/backfold/llms.txt`).
- **Drop-in coding-agent skill:** [`skills/backfold/SKILL.md`](skills/backfold/SKILL.md) (works with Claude Code, Cursor, Codex, OpenCode, and MCP-compatible agents).
- **Technical deep-dive:** [`docs/SUBNORMAL_KALMAN.md`](docs/SUBNORMAL_KALMAN.md) (x86-64 subnormal traps, Anchor-Centered Gaussian Potentials, zero-cancellation changepoint Bayes factors, 106-bit TwoSum moments, and Shiryaev–Roberts e-values).
- **Reproduce all tests & benchmarks:**
  ```sh
  cargo test --package backfold-core --package backfold-cli --package backfold-wasm --release
  cargo clippy --workspace --all-targets -- -D warnings
  pytest -v tests/
  python -m benchmarks.run_benchmarks
  ```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
