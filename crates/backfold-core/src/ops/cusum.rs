//! Exact tropical-semiring two-sided Page CUSUM monoid with O(log T) first-crossing search.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_finite, BackfoldError, Monoid};

/// One-sided Page CUSUM tropical semiring summary over a chronological interval.
///
/// Tracks:
/// - `sum`: total increment sum over the interval.
/// - `max_pref`: **non-empty** maximum prefix sum (may be negative!) so boundary crossings
///   and monotone prefix binary search never false-trigger on an empty prefix.
/// - `max_suff`: `max(0, max non-empty suffix sum)` — equals exact `S_T` starting from `S_0 = 0`.
/// - `max_sub`: `max(0, max subarray sum)` — equals exact `max_{t in I} S_t` starting from `S_0 = 0`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CusumSide {
    pub sum: f64,
    pub max_pref: f64,
    pub max_pref_ts: i64,
    pub max_suff: f64,
    pub max_suff_start_ts: i64,
    pub max_sub: f64,
    pub best_start_ts: i64,
    pub best_peak_ts: i64,
}

impl CusumSide {
    #[inline]
    pub fn identity() -> Self {
        Self {
            sum: 0.0,
            max_pref: f64::NEG_INFINITY,
            max_pref_ts: i64::MIN,
            max_suff: 0.0,
            max_suff_start_ts: i64::MIN,
            max_sub: 0.0,
            best_start_ts: i64::MIN,
            best_peak_ts: i64::MIN,
        }
    }

    #[inline]
    pub fn singleton(ts: i64, z: f64) -> Self {
        let pos_z = z.max(0.0);
        let active_ts = if pos_z > 0.0 { ts } else { i64::MIN };
        Self {
            sum: z,
            max_pref: z,
            max_pref_ts: ts,
            max_suff: pos_z,
            max_suff_start_ts: active_ts,
            max_sub: pos_z,
            best_start_ts: active_ts,
            best_peak_ts: active_ts,
        }
    }

    #[inline]
    pub fn combine(l: &Self, r: &Self) -> Self {
        let sum = l.sum + r.sum;

        // Non-empty max prefix (earliest timestamp wins ties)
        let cross_pref = l.sum + r.max_pref;
        let (max_pref, max_pref_ts) = if cross_pref > l.max_pref {
            (cross_pref, r.max_pref_ts)
        } else {
            (l.max_pref, l.max_pref_ts)
        };

        // Max suffix >= 0
        let left_ext = l.max_suff + r.sum;
        let (max_suff, max_suff_start_ts) = if l.max_suff > 0.0 && left_ext > r.max_suff && left_ext > 0.0 {
            (left_ext, l.max_suff_start_ts)
        } else if r.max_suff > 0.0 {
            (r.max_suff, r.max_suff_start_ts)
        } else {
            (0.0, i64::MIN)
        };

        // Max subarray >= 0 (left, crossing, right)
        let mut max_sub = l.max_sub;
        let mut best_start_ts = l.best_start_ts;
        let mut best_peak_ts = l.best_peak_ts;

        if l.max_suff > 0.0 {
            let cross = l.max_suff + r.max_pref;
            if cross > max_sub {
                max_sub = cross;
                best_start_ts = l.max_suff_start_ts;
                best_peak_ts = r.max_pref_ts;
            }
        }
        if r.max_sub > max_sub {
            max_sub = r.max_sub;
            best_start_ts = r.best_start_ts;
            best_peak_ts = r.best_peak_ts;
        }

        Self {
            sum,
            max_pref,
            max_pref_ts,
            max_suff,
            max_suff_start_ts,
            max_sub,
            best_start_ts,
            best_peak_ts,
        }
    }
}

/// Two-sided Page CUSUM associative monoid.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CusumMonoid {
    pub n: u64,
    pub pos: CusumSide,
    pub neg: CusumSide,
}

impl CusumMonoid {
    pub fn singleton(
        ts: i64,
        x: f64,
        mu0: f64,
        sigma0: f64,
        slack_k: f64,
    ) -> Result<Self, BackfoldError> {
        let x = check_finite(x, "x")?;
        let mu0 = check_finite(mu0, "mu0")?;
        let sigma0 = check_finite(sigma0, "sigma0")?;
        let slack_k = check_finite(slack_k, "slack_k")?;
        if sigma0 <= 0.0 {
            return Err(BackfoldError::InvalidParameter("sigma0 must be > 0"));
        }
        if slack_k < 0.0 {
            return Err(BackfoldError::InvalidParameter("slack_k must be >= 0"));
        }
        let z = (x - mu0) / sigma0;
        Self::singleton_z(ts, z, slack_k)
    }

    pub fn singleton_z(ts: i64, z: f64, slack_k: f64) -> Result<Self, BackfoldError> {
        let z = check_finite(z, "z")?;
        let slack_k = check_finite(slack_k, "slack_k")?;
        if slack_k < 0.0 {
            return Err(BackfoldError::InvalidParameter("slack_k must be >= 0"));
        }
        Ok(Self {
            n: 1,
            pos: CusumSide::singleton(ts, z - slack_k),
            neg: CusumSide::singleton(ts, -z - slack_k),
        })
    }

    #[inline]
    pub fn current_pos(&self) -> f64 {
        self.pos.max_suff
    }

    #[inline]
    pub fn current_neg(&self) -> f64 {
        self.neg.max_suff
    }

    #[inline]
    pub fn peak_pos(&self) -> f64 {
        self.pos.max_sub
    }

    #[inline]
    pub fn peak_neg(&self) -> f64 {
        self.neg.max_sub
    }

    #[inline]
    pub fn max_peak(&self) -> f64 {
        self.pos.max_sub.max(self.neg.max_sub)
    }

    #[inline]
    pub fn alarmed(&self, threshold: f64) -> bool {
        self.max_peak() >= threshold
    }

    #[inline]
    pub fn onset_ts(&self) -> i64 {
        if self.pos.max_sub >= self.neg.max_sub {
            self.pos.best_start_ts
        } else {
            self.neg.best_start_ts
        }
    }

    #[inline]
    pub fn peak_ts(&self) -> i64 {
        if self.pos.max_sub >= self.neg.max_sub {
            self.pos.best_peak_ts
        } else {
            self.neg.best_peak_ts
        }
    }
}

impl Monoid for CusumMonoid {
    type Ctx = ();

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            n: 0,
            pos: CusumSide::identity(),
            neg: CusumSide::identity(),
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(left: &Self, right: &Self, _ctx: &Self::Ctx) -> Self {
        if left.n == 0 {
            return right.clone();
        }
        if right.n == 0 {
            return left.clone();
        }
        Self {
            n: left.n + right.n,
            pos: CusumSide::combine(&left.pos, &right.pos),
            neg: CusumSide::combine(&left.neg, &right.neg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cusum_matches_sequential_for_arbitrary_parenthesization() {
        let zs = [-0.2, 0.4, -1.5, 2.2, 2.5, -0.3, 1.8, -3.5, -2.8, -2.0, 0.1];
        let k = 0.5;
        let elems: Vec<_> = zs
            .iter()
            .enumerate()
            .map(|(i, &z)| CusumMonoid::singleton_z(i as i64, z, k).unwrap())
            .collect();

        let left = elems
            .iter()
            .fold(CusumMonoid::identity(&()), |a, b| CusumMonoid::combine(&a, b, &()));
        let right = elems
            .iter()
            .rev()
            .fold(CusumMonoid::identity(&()), |b, a| CusumMonoid::combine(a, &b, &()));

        let mut sp = 0.0f64;
        let mut sn = 0.0f64;
        let mut pkp = 0.0f64;
        let mut pkn = 0.0f64;
        for &z in &zs {
            sp = (sp + z - k).max(0.0);
            sn = (sn - z - k).max(0.0);
            pkp = pkp.max(sp);
            pkn = pkn.max(sn);
        }

        assert!((left.current_pos() - sp).abs() < 1e-13);
        assert!((left.current_neg() - sn).abs() < 1e-13);
        assert!((left.peak_pos() - pkp).abs() < 1e-13);
        assert!((left.peak_neg() - pkn).abs() < 1e-13);
        assert!((left.peak_pos() - right.peak_pos()).abs() < 1e-13);
        assert!((left.peak_neg() - right.peak_neg()).abs() < 1e-13);
    }
}
