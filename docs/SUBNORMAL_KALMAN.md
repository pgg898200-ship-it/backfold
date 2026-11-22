# How x86-64 Subnormal Traps Slowed Our Exact 2D Kalman Filter by 10.9× (And Reaching 4,297× Over `-O3` Sequential Rust)

When we built [`backfold`](../README.md) — a persistent $O(\log T)$ associative-monoid engine for out-of-order, retractable, forkable streaming inference in `#![forbid(unsafe_code)]` Rust — we hit three genuinely surprising numerical and systems traps. Every single one showed up immediately because our benchmark harness asserts **exact mathematical parity (`max |diff| <= 1e-12`) against independent sequential oracles** on every run.

Here is the real engineering post-mortem of all three.

---

## 1. The Mystery: Why Did an $O(\log T)$ 2D Kalman Filter Slow Down 8× Between `T = 1k` and `T = 100k`?

Särkkä & García-Fernández (IEEE TAC, 2020) showed that linear Gaussian Kalman filtering is an **exact associative monoid** over a 5-tuple $(A, b, C, \eta, J)$ representing the affine transition and backward information potential across any chronological time interval. Put those 5-tuples inside a persistent copy-on-write B+ tree (`FoldTree<M, B>`), and repairing a late sensor reading at tick `t = 33,333` out of `T = 100,000` takes $O(\log T)$ combines instead of $O(T)$ sequential replay.

Our first implementation (`KalmanMonoid` with `Vec<f64>` fields) verified exact parity (`max_err = 3.05e-16`), and already beat handwritten `-O3` sequential Rust by **76.7×** at `T = 100,000` (`29.00 µs` vs. `2,224.6 µs`).

Then we wrote a zero-heap-allocation stack-only [`KalmanFilter2D`](../crates/backfold-core/src/ops/kalman.rs) (`[f64; 4]` / `[f64; 2]`, `Copy`, 0 `malloc`s). At `T = 1,000`, latency collapsed from `14.97 µs → 1.59 µs` (**9.4× faster**). We expected `T = 100,000` (+2 tree levels!) to sit around `2.5 µs`.

Instead, we measured:
- `T = 1,000` (depth 4): **`1.59 µs`**
- `T = 10,000` (depth 5): **`5.35 µs`** (jumped 3.4× for +1 level?!)
- `T = 100,000` (depth 6): **`12.80 µs`** (jumped another 2.4×?!)
- `T = 500,000` (depth 7): **`13.09 µs`** (barely moved?!)

That curve is **not** cache miss latency — L3 cache misses don't plateau sharply between `100k` and `500k` while jumping 8× between `1k` and `100k`.

### The Culprit: Closed-Loop Spectral Decay Into IEEE-754 Subnormals

In a stable constant-velocity Kalman filter (`q_accel = 1e-3`, `r_obs = 0.25`), the single-step closed-loop transition matrix $A = (I - K H) F$ has spectral radius $\rho(A) \approx 0.85 < 1$.

Across a tree subtree spanning $k$ ticks, the monoid's composite transition matrix scales like $A^k$:
- At $k = 1{,}000$: $0.85^{1000} \approx 10^{-70}$ — ordinary IEEE-754 normal `f64` (**1 cycle** multiply).
- At $k \approx 5{,}000$: $0.85^{5000} \approx 10^{-350}$ — **right inside the IEEE-754 subnormal band ($5 \times 10^{-324}$ to $2.2 \times 10^{-308}$)!**
- At $k > 10{,}000$: underflows cleanly to exact `0.0` again (which is why `100k → 500k` plateaued!).

On x86-64 CPUs without FTZ/DAZ enabled globally, every subnormal `f64` multiply triggers a **~150–250 cycle hardware microcode assist trap**. Internal tree nodes covering `k ≈ 4,000..8,000` ticks were sitting directly in the subnormal death zone!

### The 3-Line Fix (100% Safe Rust, Zero Global MXCSR Mutation)

Flushing entries below `1e-290` to exact `0.0` at the end of `KalmanFilter2D::combine` (8 orders of magnitude above `f64::MIN_POSITIVE`, zero impact on 16-digit `f64` accuracy) plus switching heavy monoids to the theoretically optimal branching factor `B = 4` (minimizing $B \log_B T = \frac{B}{\ln B}\ln T$):

| Iteration (`T = 100,000` 2D CV Kalman late edit, Rust) | Latency | vs. `-O3` Seq Rust | Max `\|diff\|` |
|---|---:|---:|---:|
| v0: Dynamic `Vec<f64>` `KalmanMonoid` | `29.00 µs` | `76.7×` | `3.05e-16` |
| v1: Stack-only `KalmanFilter2D` (0 heap allocs) | `12.80 µs` | `171.6×` | `2.36e-16` |
| **v2: Subnormal flush (`< 1e-290 → 0.0`) + `B = 4`** | **`2.67 µs`** | **`847.4×`** | **`3.05e-16`** |
| **v2 at `T = 500,000`** | **`2.91 µs`** | **`4,297.2×`** | **`1.67e-16`** |

And from Python (`backfold.KalmanStream(dim=2)`), repairing tick `33,333` out of `100,000` takes **`8.72 µs`** vs. `4.47 seconds` in sequential NumPy (**512,577× faster**, `max |diff| = 9.09e-13`).

---

## 2. Why Standard Chan–Pébay Variance Still Loses Digits at `1e12 + N(0, 1)`

Everyone knows naive $\frac{1}{n}\sum x_i^2 - \bar{x}^2$ suffers catastrophic cancellation on high-offset streams like Unix-microsecond timestamps or `1e12 + N(0, 1)`. Centered Pébay (2008) / Chan–Golub–LeVeque parallel merging is much better — **until you check variance error at `1e12` to `<1e-10`!**

Because `1e12 ≈ 2^40`, an IEEE-754 `f64` only has 13 bits below the binary point (`ulp(1e12) ≈ 1.22e-4`). Storing `mean` as a single `f64` throws away the sub-ULP fractional part of every subtree mean before computing `delta = b.mean - a.mean`.

Adding a single 8-byte **Knuth TwoSum** compensation term (`mean_lo: f64`) gives **106-bit effective mean precision** inside [`MomentsMonoid`](../crates/backfold-core/src/ops/moments.rs). Our adversarial Breaker test now verifies `< 1e-12` variance error on `1e12 + N(0, 1)` streams — even after inserting a `1e150` spike and retracting it.

---

## 3. Why Page e-CUSUM Breaks Multi-Stream e-BH (And the Associative Shiryaev–Roberts Semiring)

Wang & Ramdas (2022) **e-BH** guarantees $\text{FDR} \le \alpha$ across $M$ streams under **arbitrary cross-stream dependence**, provided every stream supplies a valid **e-value** ($\mathbb{E}_0[e] \le 1$).

In our first live 500-channel firehose run (10 true shifts + 50 3×-louder noise traps, $\alpha = 0.05$), feeding Page e-CUSUM's running maximum $\sup_{t \le T} M_t$ directly into e-BH yielded **25 false positives (`FDR = 0.714`)**! Why? Because Page e-CUSUM guarantees **Average Run Length** ($\text{ARL}_0 \ge 1/\alpha$, with $\mathbb{E}_0[M_T] \le T$), **not** $\mathbb{E}_0[M_T] \le 1$.

The clean algebraic fix: the **Shiryaev–Roberts** statistic $R_T = \sum_{s=1}^T \prod_{i=s}^T E_i$ is an **exact associative affine monoid in the log-sum-exp semiring**! Because every suffix product has $\mathbb{E}_0[\prod_{i=s}^T E_i] \le 1$:

$$\mathbb{E}_0\!\left[\frac{R_T}{T}\right] = \frac{1}{T} \sum_{s=1}^T \mathbb{E}_0\!\left[\prod_{i=s}^T E_i\right] \le 1 \quad \text{and} \quad \mathbb{E}_0\!\left[\frac{\sum_{s \le e} \prod_{i=s}^e E_i}{T(T+1)/2}\right] \le 1$$

Combined with a composite-null margin (`singleton_z_composite(ts, z, null_margin)`) to absorb warm-up median estimation error, our 500-channel firehose dropped immediately from `FDR = 0.714` to **`TP = 10/10, FP = 0, loud-trap FP = 0, FDR = 0.000`** — and passed 200-trial Monte Carlo FDR tests under Gaussian, Student-$t_3$, and **infinite-variance Cauchy noise**.

---

## 4. Honest Crossover: When You Should NOT Use `backfold`

If your stream is strictly append-only, strictly in-order, never retracts or edits past events, never runs arbitrary `[t1, t2]` range queries, and never branches counterfactual timelines, a plain sequential pass is faster on initial construction. On `MomentsMonoid`, bulk tree construction (`12.2 ms` @ `100k`, `53.7 ms` @ `500k`) breaks even against a single-pass `-O3` Rust loop (`0.32 ms` / `1.57 ms`) after **34–39 late edits or range queries**.

Reproduce every number in this post locally in <10 seconds:
```sh
cargo run --release --bin backfold -- demo
cargo run --release --bin backfold -- bench
```
