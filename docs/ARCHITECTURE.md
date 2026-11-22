# `backfold` Mathematical & Systems Architecture

## 1. Why Streaming Statistics Breaks on Real Streams

Every mainstream online ML / streaming statistics library (`river`, `pandas.rolling`, `polars`, sequential Kalman/RLS filters) assumes a **strictly append-only, single-timeline** stream. In production telemetry, financial ticks, IoT sensors, delayed-label pipelines, and multi-agent AI swarms, four realities break that assumption:

1. **Out-of-Order & Late Arrivals:** A packet or delayed label from timestamp $t \ll T$ arrives late. A sequential stateful filter must replay $O(T - t)$ events from $t$ to $T$.
2. **Retroactive Corrections & Retractions:** Bad sensor readings, restated fundamentals, or poisoned agent observations must be edited or retracted without corrupting downstream state.
3. **Arbitrary Time-Range & Point-in-Time Queries:** Agents and backtesters query `[t1, t2]` windows or strict `as_of(knowledge_time)` snapshots.
4. **Multi-Agent Counterfactual Branching & Echo Loops:** $W$ concurrent agents explore competing hypotheses (requiring isolated forks of a 1M-tick stream) and merge overlapping observations without last-write-wins overwrite churn or echo double-counting.

## 2. Core Data Structure: Persistent Copy-on-Write Monoid B+ Tree (`FoldTree<M, B>`)

Every event is keyed by a totally ordered causal tuple:

$$\text{EventKey} = (\text{event\_time}: \text{i64},\; \text{source\_id}: \text{u64},\; \text{seq}: \text{u64})$$

Because identical `(event_time, source_id, seq)` tuples map to the exact same leaf slot, re-inserting or gossiping the same observation across agents is strictly **idempotent** (zero echo double-counting).

Each tree node stores `Arc<Node<M, B>>` with cached associative summary `summary: M`, element count `len`, and key bounds `[min_key, max_key]`:

- **Single-owner writes (`Arc::make_mut`):** Zero cloning, in-place $O(\log_B T)$ update.
- **Persistent fork (`tree.fork()`):** $O(1)$ (`Arc::clone` of root — ~15 ns, 0 bytes copied).
- **Shared-tree writes:** Copy-on-write ONLY along the $O(\log_B T)$ root-to-leaf path (~6 nodes at $T = 10^6$).
- **Bulk ingest (`from_sorted`):** Bottom-up $O(T)$ linear construction with minimal tree height.
- **CRDT Merge (`tree_a.merge(&tree_b)`):** Associative, idempotent union with `Arc::ptr_eq` subtree short-circuiting so forked agents sharing 99.9% of history skip unchanged subtrees.
- **Monotone Prefix Binary Search (`find_first_prefix`):** $O(\log_B T)$ top-down descent carrying accumulated prefix summary `acc`, locating the exact earliest timestamp where a monotone statistic (e.g., Page CUSUM peak or e-detector) crosses a threshold $h$.

## 3. The Six Exact Associative Monoids

| Monoid | Mathematical Formulation | Associative Combine Cost | Numerical Hardening |
|---|---|---|---|
| **`MomentsMonoid`** | Chan–Golub–LeVeque / Pébay (2008) centered $(n, \mu, \mu_{\text{lo}}, M_2, M_3, M_4, \min, \max)$ | $O(1)$ (~15 ns) | Knuth TwoSum 106-bit double-double mean; `<1e-12` variance error even on $10^{12} + \mathcal{N}(0,1)$ |
| **`EwmaMonoid`** | Continuous-time irregular-$\Delta t$ exponentially weighted mean & variance | $O(1)$ (~12 ns) | Subnormal-safe decay clamp ($\lambda \Delta t > 700 \Rightarrow d = 0$) |
| **`CusumMonoid`** | Two-sided Page CUSUM tropical semiring $(\Sigma, P_{\text{nonempty}}, M_{\text{suff}}, B_{\text{sub}})$ | $O(1)$ (~10 ns) | Non-empty prefix max guarantees exact $O(\log T)$ first-alarm binary descent |
| **`EProcessMonoid`** | Waudby-Smith–Ramdas 8-point betting martingale + Page e-CUSUM + log-sum-exp Shiryaev–Roberts & all-subinterval scan | $O(K)$ ($K=8$) | Bounded odd $\tanh(z/2)$ score + composite-null margin + Wang–Ramdas `e-BH` ($\mathbb{E}_0[e] \le 1$, Cauchy-safe) |
| **`KalmanFilter1D` / `KalmanFilter2D` / `KalmanMonoid`** | Särkkä & García-Fernández (2020, IEEE TAC) 5-tuple $(A, b, C, \eta, J)$ associative Gaussian filter | $O(D^3)$ (zero-alloc stack for $D \in \{1,2\}$) | Joseph-form PSD leaf updates, symmetrization, x86-64 subnormal flush (`|x| < 1e-290 => 0`) |
| **`RidgeMonoid`** | Continuous-time discounted Gram $(X^\top W X, X^\top W y, y^\top W y)$ | $O(D^2)$ in-place `combine_assign` (inline stack for $D \le 8$) | Jacobi equilibration $D^{-1/2}(G+\alpha I)D^{-1/2}$ + pivoted LU + relative-Tikhonov rank-deficient fallback |

## 4. Top-Down `<350`-Token Lossless-Recoverable Agent Digest

Instead of scanning raw rows or truncating streams for an LLM context window, `backfold digest` walks the already-computed monoid tree top-down in $O(K \log T)$:
1. Computes global robust baseline statistics, exact CUSUM changepoint onsets/peaks, first-crossing alarm timestamps, anytime-valid e-detector p-values, and Kalman level/velocity state.
2. Uses a priority-queue top-down descent (`top_k_intervals`) to isolate the highest-surprise subintervals (rare bursts, regime transitions, outlier spikes) without averaging them away.
3. Emits a compact Markdown or JSON digest (<350 tokens) carrying exact `[t_start, t_end]` drilldown cursors so any agent can recover the exact original ticks via `backfold drill` in $O(\log T)$.
