---
name: backfold
description: Persistent O(log T) out-of-order, retractable, forkable online statistical inference (Kalman filter + exact RTS smoother + Bayesian changepoint shock locator, 106-bit multivariate Covariance/PCA/Mahalanobis, CUSUM, EWMA, Ridge, anytime e-values + e-BH FDR), native stdio MCP server (`backfold mcp`), and lossless-recoverable <300-token stream compression for AI agents. Use whenever inspecting large CSV/time-series telemetry streams, repairing late-arriving or retracted stream events, running RTS smoothing, streaming PCA, or multi-agent counterfactual simulations, or preventing multi-stream false-alarm / echo-inflation failures.
---

# backfold — Agent Streaming, MCP & Telemetry Skill (`v0.3.0`)

Use `backfold` (`pip install backfold` or native `backfold` binary) instead of dumping raw CSV / time-series rows into your context window or re-scanning history in Python/NumPy/Pandas loops.

## Rule 1: Connect `backfold mcp` or run `backfold digest` — never dump large telemetry CSVs raw
Add the zero-dependency native MCP server to Claude Desktop / Cursor / Windsurf / Claude Code:
```json
{ "mcpServers": { "backfold": { "command": "backfold", "args": ["mcp"] } } }
```
It exposes 5 tools: `backfold_digest_csv`, `backfold_drill_csv`, `backfold_kalman_rts_csv` (with automatic shock changepoint localization), `backfold_covariance_pca_csv`, and `backfold_ebh_sift_csv`.

From shell, when given any time-series CSV (`>500` rows), run `backfold digest` first:
```sh
backfold digest <path/to/stream.csv> --value-col 1 --top-k 5
```
This compresses `20,000–100,000` ticks into a **~239–256 token** pure-Markdown causal digest (**469×–2,510× compression**):
- Preserves **100% exact-window recall** on rare 5-tick acute bursts and sustained regime shifts.
- Uses causal warm-up `median + MAD` scale + composite-null Shiryaev-Roberts e-values so **3×-louder distractor channels cause 0% false alarms**.
- Recover exact raw rows losslessly for any flagged `[t_start..t_end]` cursor with:
```sh
backfold drill <path/to/stream.csv> --start <t_start> --end <t_end>
```

## Rule 2: Use `backfold` Python bindings for out-of-order, retractable, or forkable streaming inference
All bulk calls release the Python GIL (`py.allow_threads`) and accept zero-copy NumPy buffers:

| Task | `backfold` API | Verified Latency (`T=100k` late edit) | Exact Speedup |
|---|---|---|---|
| Exact RTS Kalman smoother + `log p(y)` + changepoint locator | `bf.KalmanStream(dim=2)` (`.smooth_at`, `.log_marginal_likelihood`, `.locate_changepoint`) | **11.12 µs** Py (`4.90 µs` Rust) | **578,868×** vs NumPy RTS (`3,694×` vs `-O3` Rust @500k) |
| 1D / 2D / ND Kalman filter (irregular `dt`, missing obs) | `bf.KalmanStream(dim=2)` | **8.82 µs** Py (`2.88 µs` Rust) | **517,710×** vs NumPy seq (`3,538×` vs `-O3` Rust @500k) |
| Multivariate covariance, cyclic-Jacobi PCA & Mahalanobis (`1e12`-safe) | `bf.CovarianceStream(dim=8)` (`.pca`, `.range_pca`, `.mahalanobis_many`) | **15.60 µs** Py (`5.68 µs` Rust) | **358×** vs NumPy `cov` (`262×` vs `-O3` Rust) |
| Two-sided Page CUSUM + exact earliest alarm | `bf.CusumStream(mu0, sigma0, slack_k)` | **3.83 µs** Py (`1.11 µs` Rust) | **35,897×** vs Py seq (`2,177×` vs `-O3` Rust @500k) |
| Continuous-time irregular-`dt` EWMA | `bf.EwmaStream(half_life=...)` | **3.92 µs** Py (`0.72 µs` Rust) | **495×** NumPy · **1,231×** Pandas (**`8,152×`** Rust @500k) |
| Continuous-time discounted Ridge / RLS (delayed labels) | `bf.RidgeStream(dim=8, decay_rate=...)` | **16.95 µs** Py (`6.11 µs` Rust) | **328×** NumPy · **349×** SciPy · **7,092×** River@10k |
| Higher-order moments (`1e12` offset-safe) | `bf.MomentsStream()` | **6.06 µs** Py (`1.17 µs` Rust) | **121×** NumPy · **213×** Pandas · **45×** Polars (**`1,288×`** Rust @500k) |
| Single-call multi-stream FDR sifter (`alpha = 0.05`) | `bf.sift_streams(channels_2d, warmup=120, alpha=0.05)` | **107.4 ms** on 500 channels (`FDR=0.000, Recall=1.000`) | Arbitrary-dependence Wang-Ramdas e-BH guarantee |
| Echo-proof multi-agent evidence fusion | `bf.SwarmEProcessLedger()` | 0% overwrite churn, 0% echo inflation | Eliminates blackboard overwrites & gossip double-counting |
| Zero-lookahead point-in-time backtesting | `bf.BitemporalMoments()` | $O(\log V)$ `as_of(knowledge_time)` | 0% lookahead leakage |

## Rule 3: Know when NOT to use `backfold`
- Pure append-only, single-pass, in-order aggregation with zero retractions, zero range queries, and zero forks: use a single sequential pass (break-even is ~31–33 late updates or range queries at `T = 100k–500k`).
- High-dimensional Ridge or Covariance (`d > 64`): both store $O(d^2)$ packed matrices per node (optimized for `d <= 32`, zero-alloc inline for `d <= 8`).

## Reference
See `https://pgg898200-ship-it.github.io/backfold/llms-full.txt` or `llms-full.txt` in the `backfold` repository for the complete copy-pasteable Python, Rust, CLI, and MCP API.
