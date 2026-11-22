//! Anytime-valid Shin-Ramdas-Rinaldo (2024) e-CUSUM detector, Waudby-Smith-Ramdas (2024)
//! betting martingale mixture, and Wang-Ramdas (2022) e-BH multi-stream FDR control.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_finite, BackfoldError, Monoid};

/// Symmetric 8-point geometric betting grid covering small, medium, and large shifts in either sign.
/// Because `|lambda_k| <= 0.5` and `|g_t| <= 1.0`, `1 + lambda_k * g_t >= 0.5 > 0` always holds.
pub const BETTING_GRID: [f64; 8] = [-0.5, -0.25, -0.1, -0.04, 0.04, 0.1, 0.25, 0.5];

const LN_K: f64 = 2.0794415416798357; // ln(8.0)

#[inline]
fn logmeanexp8(v: &[f64; 8]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in v {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &x in v {
        sum += (x - m).exp();
    }
    m + sum.ln() - LN_K
}

#[inline]
fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// Associative monoid maintaining THREE exact sequential-testing operators simultaneously:
/// 1. Cumulative Waudby-Smith-Ramdas betting martingale mixture (`cum_log_w[k]`, `E_0[W_T] <= 1`).
/// 2. Exact Shin-Ramdas-Rinaldo Page e-CUSUM tropical-semiring detector (`edet_max_suff[k]`, `ARL_0 >= 1/alpha`).
/// 3. Exact log-sum-exp Shiryaev-Roberts suffix (`sr_suff_log[k]`) and all-subinterval scan
///    (`sr_sub_log[k]`) associative semiring yielding provably valid changepoint e-values
///    (`E_0[SR_T / T] <= 1` and `E_0[Scan_T / (T(T+1)/2)] <= 1`) for arbitrary-dependence e-BH FDR control.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EProcessMonoid {
    pub n: u64,
    pub cum_log_w: [f64; 8],
    pub edet_max_pref: [f64; 8],
    pub edet_max_suff: [f64; 8],
    pub edet_max_sub: [f64; 8],
    pub sr_pref_log: [f64; 8],
    pub sr_suff_log: [f64; 8],
    pub sr_sub_log: [f64; 8],
    pub peak_ts: i64,
    pub peak_val: f64,
}

impl EProcessMonoid {
    fn from_bounded_score_with_margin(ts: i64, g: f64, null_margin: f64) -> Self {
        let mut cum_log_w = [0.0f64; 8];
        let mut edet_max_pref = [0.0f64; 8];
        let mut edet_max_suff = [0.0f64; 8];
        let mut edet_max_sub = [0.0f64; 8];
        let mut sr_pref_log = [0.0f64; 8];
        let mut sr_suff_log = [0.0f64; 8];
        let mut sr_sub_log = [0.0f64; 8];
        let mut best_sub = 0.0f64;
        for k in 0..8 {
            let lam = BETTING_GRID[k];
            // Composite null |E[g]| <= null_margin: shift against each bet's direction.
            let g_adj = if lam > 0.0 {
                g - null_margin
            } else {
                g + null_margin
            };
            let inc = (lam * g_adj).max(-0.999_999).ln_1p();
            cum_log_w[k] = inc;
            edet_max_pref[k] = inc;
            let pos = inc.max(0.0);
            edet_max_suff[k] = pos;
            edet_max_sub[k] = pos;
            sr_pref_log[k] = inc;
            sr_suff_log[k] = inc;
            sr_sub_log[k] = inc;
            if pos > best_sub {
                best_sub = pos;
            }
        }
        Self {
            n: 1,
            cum_log_w,
            edet_max_pref,
            edet_max_suff,
            edet_max_sub,
            sr_pref_log,
            sr_suff_log,
            sr_sub_log,
            peak_ts: if best_sub > 0.0 { ts } else { i64::MIN },
            peak_val: best_sub,
        }
    }

    /// Exact bounded-observation betting martingale leaf (`x in [0, 1]`, null mean `mu0 in [0, 1]`).
    pub fn singleton_bounded(ts: i64, x: f64, mu0: f64) -> Result<Self, BackfoldError> {
        let x = check_finite(x, "x")?;
        let mu0 = check_finite(mu0, "mu0")?;
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&mu0) {
            return Err(BackfoldError::InvalidParameter(
                "x and mu0 must lie in [0, 1]",
            ));
        }
        Ok(Self::from_bounded_score_with_margin(ts, x - mu0, 0.0))
    }

    /// Distribution-free symmetric-null betting martingale leaf via odd bounded score `tanh(z / 2)`.
    /// Valid under any zero-mean symmetric noise distribution (including heavy-tailed Cauchy/Student-t).
    pub fn singleton_z(ts: i64, z: f64) -> Result<Self, BackfoldError> {
        Self::singleton_z_composite(ts, z, 0.0)
    }

    /// Composite-null symmetric betting martingale leaf testing `H0: |E[tanh(z / 2)]| <= null_margin`.
    /// Useful when baseline center is estimated from a finite causal warm-up window (`null_margin in [0, 0.5]`).
    pub fn singleton_z_composite(
        ts: i64,
        z: f64,
        null_margin: f64,
    ) -> Result<Self, BackfoldError> {
        let z = check_finite(z, "z")?;
        let margin = check_finite(null_margin, "null_margin")?;
        if !(0.0..=0.5).contains(&margin) {
            return Err(BackfoldError::InvalidParameter(
                "null_margin must lie in [0, 0.5]",
            ));
        }
        let g = (0.5 * z).tanh();
        Ok(Self::from_bounded_score_with_margin(ts, g, margin))
    }

    /// Exact mixture log-wealth `log( (1/8) * sum_k exp(cum_log_w[k]) )`.
    #[inline]
    pub fn log_wealth(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            logmeanexp8(&self.cum_log_w)
        }
    }

    /// Exact mixture wealth `W_T` (valid e-value `E_0[W_T] <= 1` at any stopping time `T`).
    #[inline]
    pub fn wealth(&self) -> f64 {
        self.log_wealth().min(700.0).exp()
    }

    /// Exact Page e-CUSUM mixture log-detector at interval end `T`.
    #[inline]
    pub fn log_e_detector(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            logmeanexp8(&self.edet_max_suff)
        }
    }

    /// Exact Page e-CUSUM detector value `M_T` at interval end `T` (guarantees `ARL_0 >= 1/alpha`).
    #[inline]
    pub fn e_detector(&self) -> f64 {
        self.log_e_detector().min(700.0).exp()
    }

    /// Exact Shiryaev-Roberts changepoint e-value `E_SR = (1 / (8*T)) * sum_k sum_{s=1}^T prod_{i=s}^T E_{i,k}`.
    /// Provably satisfies `E_0[E_SR] <= 1` under H0 for ANY unknown changepoint onset time `s`.
    #[inline]
    pub fn sr_evalue(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        let log_sr = logmeanexp8(&self.sr_suff_log) - (self.n as f64).ln();
        log_sr.min(700.0).exp()
    }

    /// Exact all-subinterval scan e-value `E_scan = (1 / (8 * T(T+1)/2)) * sum_k sum_{s<=e} prod_{i=s}^e E_{i,k}`.
    /// Provably satisfies `E_0[E_scan] <= 1` under H0 and catches transient mid-stream bursts.
    #[inline]
    pub fn scan_evalue(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        let nf = self.n as f64;
        let pairs = 0.5 * nf * (nf + 1.0);
        let log_scan = logmeanexp8(&self.sr_sub_log) - pairs.ln();
        log_scan.min(700.0).exp()
    }

    /// Provably valid combined e-value `(wealth + sr_evalue + scan_evalue) / 3`.
    /// Because it is a convex combination of three exact e-values, `E_0[e_value()] <= 1.0` holds
    /// mathematically for any horizon `T`, making it a drop-in input to Wang-Ramdas `ebh_reject`.
    #[inline]
    pub fn e_value(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        (self.wealth() + self.sr_evalue() + self.scan_evalue()) / 3.0
    }

    /// Alias for [`Self::e_value`]: provably valid changepoint & burst e-value (`E_0 <= 1`).
    #[inline]
    pub fn peak_e_detector_valid(&self) -> f64 {
        self.e_value()
    }

    /// Valid e-value p-value `min(1, 1 / e_value())` (`P_0(p <= alpha) <= alpha` by Markov's inequality).
    #[inline]
    pub fn anytime_p_value(&self) -> f64 {
        let ev = self.e_value().max(1.0);
        (1.0 / ev).min(1.0)
    }

    #[inline]
    pub fn rejected(&self, alpha: f64) -> bool {
        if alpha <= 0.0 || !alpha.is_finite() {
            return false;
        }
        self.e_value() >= 1.0 / alpha
    }
}

impl Monoid for EProcessMonoid {
    type Ctx = ();

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            n: 0,
            cum_log_w: [0.0; 8],
            edet_max_pref: [f64::NEG_INFINITY; 8],
            edet_max_suff: [0.0; 8],
            edet_max_sub: [0.0; 8],
            sr_pref_log: [f64::NEG_INFINITY; 8],
            sr_suff_log: [f64::NEG_INFINITY; 8],
            sr_sub_log: [f64::NEG_INFINITY; 8],
            peak_ts: i64::MIN,
            peak_val: 0.0,
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(l: &Self, r: &Self, _ctx: &Self::Ctx) -> Self {
        if l.n == 0 {
            return r.clone();
        }
        if r.n == 0 {
            return l.clone();
        }
        let mut cum_log_w = [0.0f64; 8];
        let mut edet_max_pref = [0.0f64; 8];
        let mut edet_max_suff = [0.0f64; 8];
        let mut edet_max_sub = [0.0f64; 8];
        let mut sr_pref_log = [0.0f64; 8];
        let mut sr_suff_log = [0.0f64; 8];
        let mut sr_sub_log = [0.0f64; 8];
        let mut best_sub = 0.0f64;

        for k in 0..8 {
            let lw = l.cum_log_w[k];
            let rw = r.cum_log_w[k];
            cum_log_w[k] = lw + rw;

            // Tropical semiring (Page e-CUSUM)
            edet_max_pref[k] = l.edet_max_pref[k].max(lw + r.edet_max_pref[k]);
            let left_ext = if l.edet_max_suff[k] > 0.0 {
                (l.edet_max_suff[k] + rw).max(0.0)
            } else {
                0.0
            };
            edet_max_suff[k] = left_ext.max(r.edet_max_suff[k]);
            let cross = if l.edet_max_suff[k] > 0.0 {
                (l.edet_max_suff[k] + r.edet_max_pref[k]).max(0.0)
            } else {
                0.0
            };
            let sub = l.edet_max_sub[k].max(r.edet_max_sub[k]).max(cross);
            edet_max_sub[k] = sub;
            if sub > best_sub {
                best_sub = sub;
            }

            // Log-sum-exp semiring (exact Shiryaev-Roberts + all-subinterval scan)
            sr_pref_log[k] = logaddexp(l.sr_pref_log[k], lw + r.sr_pref_log[k]);
            sr_suff_log[k] = logaddexp(l.sr_suff_log[k] + rw, r.sr_suff_log[k]);
            let cross_sub = l.sr_suff_log[k] + r.sr_pref_log[k];
            sr_sub_log[k] = logaddexp(logaddexp(l.sr_sub_log[k], r.sr_sub_log[k]), cross_sub);
        }

        let peak_ts = if r.peak_val > l.peak_val {
            r.peak_ts
        } else {
            l.peak_ts
        };

        Self {
            n: l.n + r.n,
            cum_log_w,
            edet_max_pref,
            edet_max_suff,
            edet_max_sub,
            sr_pref_log,
            sr_suff_log,
            sr_sub_log,
            peak_ts,
            peak_val: best_sub,
        }
    }
}

/// Wang & Ramdas (2022) e-BH multiple-testing procedure.
///
/// Guarantees `FDR <= alpha` under **arbitrary dependence** across streams and agents
/// (no independence assumption required). Returns sorted indices of rejected hypotheses.
pub fn ebh_reject(e_values: &[f64], alpha: f64) -> Vec<usize> {
    let m = e_values.len();
    if m == 0 || alpha <= 0.0 || !alpha.is_finite() {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f64)> = e_values
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, if e.is_finite() && e > 0.0 { e } else { 0.0 }))
        .collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mf = m as f64;
    let mut k_star = 0usize;
    for (rank_zero, &(_, e_val)) in indexed.iter().enumerate() {
        let k = rank_zero + 1;
        let threshold = mf / (alpha * (k as f64));
        if e_val >= threshold {
            k_star = k;
        }
    }
    let mut out: Vec<usize> = indexed.into_iter().take(k_star).map(|(i, _)| i).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eprocess_matches_sequential_and_detects_shift() {
        let mut zs = vec![0.0f64; 100];
        zs.extend(std::iter::repeat_n(2.0, 100));
        let elems: Vec<_> = zs
            .iter()
            .enumerate()
            .map(|(i, &z)| EProcessMonoid::singleton_z(i as i64, z).unwrap())
            .collect();
        let left = elems
            .iter()
            .fold(EProcessMonoid::identity(&()), |a, b| EProcessMonoid::combine(&a, b, &()));
        let right = elems
            .iter()
            .rev()
            .fold(EProcessMonoid::identity(&()), |b, a| EProcessMonoid::combine(a, &b, &()));

        assert!((left.log_wealth() - right.log_wealth()).abs() < 1e-12);
        assert!((left.log_e_detector() - right.log_e_detector()).abs() < 1e-12);
        assert!(left.rejected(0.01));
    }

    #[test]
    fn ebh_controls_multiple_testing() {
        let es = [0.8, 1.2, 120.0, 0.5, 95.0, 1.1, 0.9, 1.0, 0.7, 1.3];
        let rej = ebh_reject(&es, 0.05);
        // m=10, alpha=0.05 -> k=1 needs 200, k=2 needs 100 -> wait: 95 < 100 so 0 rejected!
        assert!(rej.is_empty());
        let es2 = [0.8, 1.2, 120.0, 0.5, 105.0, 1.1, 0.9, 1.0, 0.7, 1.3];
        assert_eq!(ebh_reject(&es2, 0.05), vec![2, 4]);
    }
}
