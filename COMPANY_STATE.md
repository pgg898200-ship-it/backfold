# ENGINEERING_LOG — `backfold` Iterative Optimization & Verification Journal

> Every numerical bug caught, rejected design, compounding systems optimization, and benchmark receipt is recorded here verbatim from live execution on x86-64 Linux.

## 1. Problem & Market Gap

Audited existing open-source streaming and online statistics tooling:
- `online-ml/river`: Python-loop bound, strictly append-only, no $O(\log T)$ retractions or persistent forks, no Kalman or anytime e-process operators.
- `jakorostami/expectation`: GPL-3.0, stationary i.i.d. e-values without retractable chronological trees or changepoint semirings.
- `evilrat420/irithyll` / `focus-cpt`: Rust streaming ML and convex-hull GLR without $O(\log T)$ retractions, arbitrary range folds, or multi-agent CRDT merging.

**Design thesis:** Compile exact online statistical and state-space operators into associative monoids inside a persistent copy-on-write B+ tree (`FoldTree<M, B>`), exposing a `#![forbid(unsafe_code)]` Rust crate, zero-copy PyO3 Python package, and sub-5ms native CLI under Apache-2.0.

## 2. Iterative Engineering & Scientific Breakthroughs

### Iteration 1 — Eliminating Sub-ULP Mean Cancellation at `1e12 + N(0, 1)`
- **Issue caught by test suite:** Standard Chan–Pébay centered `MomentsMonoid` with a single `f64` mean lost sub-ULP fractional bits at `1e12` offset (`ulp(1e12) ≈ 1.22e-4`), producing `~1e-4` variance drift.
- **Root fix:** Added a Knuth TwoSum error-compensation field (`mean_lo: f64`) inside `MomentsMonoid`, giving **106-bit effective mean precision** with zero heap allocations.
- **Verified result:** Variance, skewness, and kurtosis at `1e12 + N(0, 1)` match exact represented residuals to **`< 1e-12`**.

### Iteration 2 — Exact Associative Shiryaev–Roberts E-Values for Arbitrary-Dependence e-BH FDR
- **Issue caught by live 500-stream eval:** Feeding Page e-CUSUM's running maximum (`sup_t M_t`) into Wang–Ramdas `ebh_reject(alpha=0.05)` produced `FP = 25 (FDR = 0.714)`, because Page e-CUSUM guarantees `ARL_0 >= 1/alpha` (`E_0[M_T] <= T`), not `E_0[e] <= 1`. Estimating a baseline median on a finite warm-up window `W` also introduces `O(1/sqrt(W))` center bias.
- **Root fix:** Derived an **exact associative log-sum-exp semiring** inside `EProcessMonoid` simultaneously tracking:
  1. Waudby-Smith–Ramdas 8-point betting martingale `wealth()` (`E_0 <= 1`),
  2. Exact Shiryaev–Roberts suffix e-value `sr_evalue() = SR_T / T` (`E_0 <= 1`), and
  3. Exact all-subinterval scan e-value `scan_evalue() = Scan_T / (T(T+1)/2)` (`E_0 <= 1`),
  combined with a composite-null margin (`singleton_z_composite(ts, z, null_margin)` testing $|\mathbb{E}[\tanh(z/2)]| \le \delta$).
- **Verified result:** 500-stream firehose (10 true shifts + 50 3×-loud traps) dropped from `FDR = 0.714 (25 FP)` to **`TP = 10/10, FP = 0, loud-trap FP = 0, FDR = 0.000`** (vs. naive union `FDR = 0.841, 53 FP`). Passed 200-trial Monte Carlo FDR tests under Gaussian, Student-$t_3$, and infinite-variance Cauchy noise.

### Iteration 3 — 10.9× Compounding Speedup on 2D Kalman via x86-64 Subnormal Diagnosis
- **Baseline (`KalmanMonoid` with `Vec<f64>`):** `T=100k` 2D CV late edit took **`29.00 µs`** (`76.7×` vs `-O3` seq Rust).
- **Opt A (Zero-heap-alloc stack `KalmanFilter2D`):** Eliminated ~16 heap allocations per combine -> `12.80 µs` (`171.6×` at `100k`; `1.59 µs` at `1k`).
- **Opt B (x86-64 IEEE-754 subnormal microcode trap fix):** Investigated why `T=1k → 100k` jumped `8×` for only +2 tree levels. Closed-loop Kalman transition $A^k$ ($\rho(A) \approx 0.85$) decays into $10^{-350}$ subnormals around $k \approx 5{,}000$, triggering ~200-cycle x86-64 denormal microcode assists on internal tree nodes. Flushing `|x| < 1e-290 => 0.0` + tuning branching factor `B = 4` dropped `T=100k` from **`12.80 µs → 2.67 µs` (`847×` vs `-O3` seq Rust)** and `T=500k` to **`2.91 µs` (`4,297×` vs `-O3` seq Rust, `max |diff| = 1.67e-16`)**.
- **Opt C (Zero-alloc inline `RidgeMonoid` for `dim <= 8` + in-place `combine_assign` + zero-copy `from_sorted`):** Cut bulk `T=500k` tree construction from `231.3 ms → 53.7 ms` (**4.3× faster**), lowering pure append-only break-even from `144` to `34.4` late edits/range queries.

### Iteration 4 — Adversarial Breaker Hardening
- `crates/backfold-core/tests/breaker_audit.rs` caught an edge case in `RidgeMonoid::solve` where exact collinearity combined with `1e24` feature-scale mismatch caused tiny `alpha` to fall below `ulp(G_ii)`. Added an equilibrated relative-Tikhonov (`1e-12 * I` in Jacobi-scaled coordinates) fallback when `alpha > 0.0`, recovering exact minimum-norm predictions to `<1e-4`.
- Rust's type checker caught a persistent-return bug in `backfold-py` where `self.tree.merge(&other.tree)` returns a new tree instead of mutating in place; fixed uniformly across all Python stream classes.

## 3. Verified Quality Gate Status

- `cargo test --package backfold-core --package backfold-cli --release`: **11/11** unit + **7/7** Breaker audit tests passed.
- `cargo clippy --workspace --all-targets -- -D warnings`: **0 warnings**.
- `pytest -v tests/`: **12/12** passed.
- `ruff check python/ benchmarks/ tests/`: **All checks passed**.
- Native CLI cold startup: **`3.25 ms` median** (`3.40 ms` p95).
- Canonical machine-readable receipt: [`benchmarks/results.json`](benchmarks/results.json).
