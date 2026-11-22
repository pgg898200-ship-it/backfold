# backfold

[![CI](https://github.com/pgg898200-ship-it/backfold/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg898200-ship-it/backfold/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Persistent $O(\log T)$ out-of-order, retractable, forkable online statistical inference & lossless-recoverable AI agent stream compression.**

Pure safe Rust (`#![forbid(unsafe_code)]`) · Zero-copy Python bindings · 3.25 ms native CLI · [Docs & `llms.txt`](https://pgg898200-ship-it.github.io/backfold/)

---

> **Every streaming library assumes time moves forward, nothing gets retracted, and only one consumer reads the stream.**
> Real sensor pipelines, financial feeds, telemetry firehoses, and multi-agent LLM swarms break all three assumptions. When an event arrives 50,000 ticks late—or gets retracted, restated, or branched by 64 counterfactual agents—`river`, `pandas.ewm`, and sequential `numpy`/`scipy` Kalman filters fall back to **$O(T)$ full replay**.
>
> `backfold` compiles every operator into an **exact associative monoid** inside a persistent copy-on-write B+ fold tree (`FoldTree<M, B>`): **$O(\log T)$ late edits & retractions, $O(\log T)$ arbitrary range queries, $O(1)$ zero-copy `fork()` (~20 ns), echo-proof multi-agent CRDT merges, distribution-free e-BH FDR control, and `<300`-token lossless-recoverable LLM stream digests — with zero approximation (`max |diff| <= 1e-12`).**

---

## 30-Second Quickstart

### Native CLI (3.25 ms cold startup, zero dependencies)
```sh
cargo build --release --bin backfold

# Run 5 live verified 100,000-tick demos in 0.58 s (0 approximations, 0 downloads)
./target/release/backfold demo

# Compress a 100,000-row CSV (~600k tokens) into a ~239-token causal Markdown digest (2,510× compression)
./target/release/backfold digest benchmarks/sample_100k.csv

# Losslessly drill down to exact raw ticks for any flagged hotspot window
./target/release/backfold drill benchmarks/sample_100k.csv --start 22498 --end 22507

# Reproduce the live Rust O(log T) vs. handwritten -O3 sequential Rust benchmark
./target/release/backfold bench
```

### Python (`pip install` / `maturin develop --release`)
```python
import numpy as np
import backfold as bf

ts = np.arange(100_000, dtype=np.int64)
dts = np.ones(100_000)
ys = np.cumsum(np.random.default_rng(0).normal(0.05, 0.1, 100_000))

# 1. Exact 2D Constant-Velocity Kalman filter with O(log T) late-arrival repair
kf = bf.KalmanStream(dim=2)
kf.ingest_constant_velocity(ts, ys, dts, q_accel=1e-3, r_obs=0.25)

# 2. O(1) zero-copy counterfactual fork (~20 ns, 0 bytes copied) + O(log T) edit (~8.7 µs)
branch = kf.fork()
branch.upsert_constant_velocity(33_333, float(ys[33_333] + 10.0), 1.0, 1e-3, 0.25)
mean, cov = branch.posterior([0.0, 0.0], [10.0, 0.0, 0.0, 10.0])

# 3. Lossless-recoverable <300-token digest for LLM / coding agents
digest = bf.digest_stream("sensor_01", ts, ys, top_k=5, threshold=18.0)
print(digest["markdown"])  # 2,510× smaller than raw CSV, 100% acute-burst window recall
```

---

## Verified Live Benchmarks (`[LIVE-CPU]`, 2-vCPU Linux x86-64)

Every number below is measured live on this hardware (`benchmarks/results.json`) and asserts **exact numerical parity** against independent naive sequential oracles (`benchmarks/oracles.py`). Python timings conservatively measure a full **edit + query + restore** round-trip (two tree updates + one summary).

### 1. Out-of-Order Late Edit Latency — Python API (`T = 100,000`, `max |diff| <= 1e-12`)

| Operator | `backfold` (Python) | NumPy | SciPy | Pandas | Polars | River (`T=10k`) | Speedup | Max `\|diff\|` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **2D Kalman CV** | **8.72 µs** | 4,471,206 µs | — | — | — | — | **512,577× vs NumPy** | `9.09e-13` |
| **Page CUSUM + 1st alarm** | **5.32 µs** | 81,766 µs *(Py seq)* | — | — | — | — | **15,378× vs seq** | `1.78e-15` |
| **CT Irregular-$\Delta t$ EWMA** | **5.03 µs** | 2,079.5 µs | — | 5,204.4 µs | — | — | **413× NumPy · 1,034× Pandas** | `4.44e-16` |
| **8D Discounted Ridge** | **17.47 µs** | 9,303.2 µs | 9,898.6 µs | — | — | 99,306 µs *(10k)* | **533× NumPy · 567× SciPy · 6,618× River** | `1.33e-15` |
| **Compensated Moments** *(mean/var/skew/kurt)* | **10.63 µs** | 965.9 µs | — | 1,554.8 µs | 301.0 µs | 8,077 µs *(10k)* | **91× NumPy · 146× Pandas · 28× Polars** | `4.57e-16` |

### 2. Rust-vs-Rust Head-to-Head (`backfold` $O(\log T)$ vs. Handwritten `-O3` Sequential Rust)

Not just "Rust vs. Python" — even against tight handwritten `-O3` Rust loops, $O(\log T)$ associative monoids win by **two to four orders of magnitude**:

| Operator | `T = 10,000` | `T = 100,000` | `T = 500,000` | `backfold` @ 500k | Seq Rust `-O3` @ 500k | Max `\|diff\|` |
|---|---:|---:|---:|---:|---:|---:|
| **1D Kalman RW** (`KalmanFilter1D`) | 164× | 1,370× | **6,239×** | **0.97 µs** | 6,020.4 µs | `8.33e-17` |
| **CT Irregular-$\Delta t$ EWMA** | 110× | 932× | **4,487×** | **1.30 µs** | 5,840.4 µs | `2.95e-17` |
| **2D Kalman CV** (`KalmanFilter2D`) | 94× | 847× | **4,297×** | **2.91 µs** | 12,498.3 µs | `1.67e-16` |
| **Page CUSUM + 1st alarm** | 32× | 269× | **1,364×** | **1.89 µs** | 2,582.8 µs | `1.78e-15` |
| **8D Discounted Ridge** | 75× | **652×** | — | *6.08 µs @ 100k* | *3,966.5 µs @ 100k* | `2.00e-15` |
| **Compensated Moments** | 11× | 93× | **418×** | **3.75 µs** | 1,567.0 µs | `3.12e-14` |
| **Arbitrary 60% Range Query** | — | 66× | **340×** | **2.76 µs** | 936.0 µs | `9.21e-15` |

### 3. Multi-Agent Coordination, Echo Immunity & 500-Stream FDR Control

| Scenario | Naive Baseline | `backfold` | Edge |
|---|---|---|---|
| **16-Agent Gossip Swarm** *(2,000 events echoed 3× out-of-order)* | 66.7% overwrite churn (4,000 overwrites); `1.732×` ($\sqrt{3}$) echo z-inflation | **0.0% overwrite churn**; 4,000/4,000 echoes deduplicated; **1.000× exact evidence** | Echo-proof `(event_time, source_id, seq)` CRDT |
| **500-Stream Firehose** *(10 true shifts + 50 3×-loud traps, $\alpha=0.05$)* | 63 alarms: 10 TP, **53 FP (50/50 loud traps!)**, **FDR = 84.1%** | **10 alarms: 10/10 TP, 0 FP, 0/50 loud traps, FDR = 0.0%** (`279 ms` total) | Causal MAD + composite-null Shiryaev-Roberts e-value + Wang-Ramdas e-BH |
| **64-Agent Counterfactual Forking** *(`T=100k`, 5 edits/agent)* | 751.8 ms (`11.75 ms/agent`) | **1.19 ms** (`0.0186 ms/agent`, `max_diff = 3.55e-15`) | **632.4× faster** ($O(1)$ persistent `Arc` fork) |

### 4. Lossless-Recoverable AI Agent Stream Compression (`50 streams × 20,000 ticks`)

| Representation | Avg Tokens | Compression | Acute Burst Exact-Window Recall | Regime Shift Recall | Loud 3×-Distractor FP Rate | Lossless Recovery |
|---|---:|---:|---:|---:|---:|---|
| Raw CSV | 120,000 | 1× | — *(overflows context)* | — | — | Yes |
| `pandas.describe()` | 17 | 7,059× | **0.0%** *(drops timestamps)* | 100.0% | **100.0%** | No |
| Uniform 200× Downsample | 600 | 200× | **20.0%** *(misses 5-tick spikes)* | 86.7% | **30.0%** | No |
| **`backfold digest`** | **255.8** | **469.1×** *(2,510× on 100k)* | **100.0%** | **100.0%** | **0.0%** | **Yes (`backfold drill`)** |

---

## What's Inside (6 Exact Associative Operators + Swarm CRDT)

1. **`MomentsStream` / `MomentsMonoid`** — Centered Chan–Pébay–Terriberry higher-order moments (`mean`, `variance`, `skewness`, `excess_kurtosis`, `min`/`max` + exact timestamps) upgraded with **Knuth TwoSum double-double mean compensation** (`106-bit` effective mean precision: `<1e-10` variance accuracy even on `1e12 + N(0, 1)` streams where standard `np.var` loses 10+ digits).
2. **`EwmaStream` / `EwmaMonoid`** — Continuous-time irregular-$\Delta t$ centered exponentially weighted mean & variance with subnormal-safe decay clamping (`lambda * dt > 700 => 0`).
3. **`CusumStream` / `CusumMonoid`** — Two-sided tropical-semiring Page CUSUM with non-empty `max_pref` (identity = $-\infty$) supporting exact **$O(\log T)$ earliest-alarm binary descent** (`first_alarm(threshold)`).
4. **`EProcessStream` / `EProcessMonoid` + `ebh_reject`** — Simultaneously tracks three exact sequential operators in a single tree: Waudby-Smith–Ramdas 8-point symmetric betting martingale (`wealth`), Shin–Ramdas–Rinaldo Page e-CUSUM (`e_detector`), and exact log-sum-exp **Shiryaev–Roberts (`sr_evalue`) + all-subinterval scan (`scan_evalue`)** yielding a provably valid composite-null e-value ($\mathbb{E}_0[e] \le 1$, even under heavy-tailed Cauchy/Student-$t$ noise) for Wang–Ramdas arbitrary-dependence **e-BH FDR control**.
5. **`KalmanStream` / `KalmanFilter1D` / `KalmanFilter2D` / `KalmanMonoid`** — Exact Särkkä & García-Fernández (IEEE TAC 2020) associative Bayesian filter supporting irregular $\Delta t$, missing observations, scalar/vector observations, Joseph-form PSD leaves, zero-heap-allocation stack `1D`/`2D` fast paths, and x86-64 subnormal flush (`|x| < 1e-290 => 0`).
6. **`RidgeStream` / `RidgeMonoid`** — Continuous-time exponentially discounted Ridge / RLS regression with zero-alloc inline stack storage for `dim <= 8`, in-place `combine_assign`, **Jacobi-equilibrated Cholesky** ($D^{-1/2}(G + \alpha I)D^{-1/2}$) with partial-pivoted LU fallback (handling Gram condition numbers up to $10^{20}$), and exact predictive variance.
7. **`SwarmEProcessLedger` & `BitemporalMoments`** — Echo-proof multi-agent CRDT ledger (`Arc::ptr_eq`-accelerated idempotent merge) and zero-lookahead bitemporal `as_of(knowledge_time)` snapshots.

---

## Honest Crossovers & Limitations

`backfold` is **not** magic for every workload. Know where it wins and where a simpler tool wins:

- **Pure append-only, single-pass, in-order aggregation:** If you never receive late arrivals, never retract/edit history, never query arbitrary sub-intervals `[t1, t2]`, and never fork branches, a plain sequential loop is faster on initial ingest. On `MomentsMonoid` at `T = 100k–500k`, bulk tree construction takes `12.2 ms` (`T=100k`) / `53.7 ms` (`T=500k`) vs. `0.32 ms` / `1.57 ms` for one sequential Rust pass — breaking even after **34–39 late updates or range queries**.
- **High-dimensional Ridge (`d > 64`):** `RidgeMonoid` maintains packed upper-triangular Gram matrices ($O(d^2)$ per tree node). It shines for streaming telemetry / control / factor models (`d <= 32`, zero-alloc inline for `d <= 8`), not 10,000-feature dense NLP models.
- **Exact quantile / median queries:** Order statistics (arbitrary quantiles) do not admit a fixed-size exact associative monoid without sketch approximation. `backfold` sticks strictly to **exact** operators (`causal_robust_scale` computes exact median/MAD on causal warm-up prefixes).

---

## For AI Coding Agents

- **Agent discovery index:** [`llms.txt`](llms.txt) and complete API reference [`llms-full.txt`](llms-full.txt) (pure Markdown).
- **Drop-in coding-agent skill:** [`skills/backfold/SKILL.md`](skills/backfold/SKILL.md) (works with Claude Code, Cursor, Codex, OpenCode, and MCP-compatible agents).
- **Technical deep-dive:** [`docs/SUBNORMAL_KALMAN.md`](docs/SUBNORMAL_KALMAN.md) (the x86-64 subnormal diagnosis, 106-bit TwoSum moments, and Shiryaev–Roberts e-values).
- **Reproduce all tests & benchmarks:**
  ```sh
  cargo test --workspace --release
  cargo clippy --workspace --all-targets -- -D warnings
  pytest -v tests/
  python -m benchmarks.run_benchmarks
  ```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
