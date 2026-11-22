//! Exact Chan-Golub-LeVeque / Pébay (2008) / Terriberry associative higher-order moments monoid.
//!
//! Centered merging operates strictly on `delta = mean_b - mean_a`, preventing catastrophic
//! cancellation even on adversarial inputs like `1e12 + N(0, 1)`.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_finite, BackfoldError, Monoid};

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MomentsMonoid {
    pub n: u64,
    pub mean: f64,
    /// Low-order Knuth TwoSum error-compensation term for `mean`, giving 106-bit effective
    /// mean precision so variance/skew/kurtosis stay accurate to ~1e-14 even on `1e12 + N(0,1)`.
    pub mean_lo: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub min: f64,
    pub max: f64,
    pub t_min: i64,
    pub t_max: i64,
    pub t_first: i64,
    pub t_last: i64,
}

impl MomentsMonoid {
    pub fn singleton(ts: i64, x: f64) -> Result<Self, BackfoldError> {
        let x = check_finite(x, "x")?;
        Ok(Self {
            n: 1,
            mean: x,
            mean_lo: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: x,
            max: x,
            t_min: ts,
            t_max: ts,
            t_first: ts,
            t_last: ts,
        })
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            (self.m2 / (self.n as f64)).max(0.0)
        }
    }

    #[inline]
    pub fn sample_variance(&self) -> f64 {
        if self.n <= 1 {
            0.0
        } else {
            (self.m2 / ((self.n - 1) as f64)).max(0.0)
        }
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline]
    pub fn sample_std(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    #[inline]
    pub fn skewness(&self) -> f64 {
        if self.n == 0 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        let var = self.m2 / n;
        (self.m3 / n) / (var * var.sqrt())
    }

    #[inline]
    pub fn excess_kurtosis(&self) -> f64 {
        if self.n == 0 || self.m2 <= 0.0 {
            return 0.0;
        }
        let n = self.n as f64;
        let var = self.m2 / n;
        (self.m4 / n) / (var * var) - 3.0
    }
}

impl Monoid for MomentsMonoid {
    type Ctx = ();

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            n: 0,
            mean: 0.0,
            mean_lo: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            t_min: i64::MAX,
            t_max: i64::MAX,
            t_first: i64::MAX,
            t_last: i64::MIN,
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(a: &Self, b: &Self, _ctx: &Self::Ctx) -> Self {
        if a.n == 0 {
            return b.clone();
        }
        if b.n == 0 {
            return a.clone();
        }
        let na = a.n as f64;
        let nb = b.n as f64;
        let n_u64 = a.n + b.n;
        let n = n_u64 as f64;

        // Error-free double-double difference: delta = (b.mean + b.mean_lo) - (a.mean + a.mean_lo)
        let (d_hi, d_err) = two_sum(b.mean, -a.mean);
        let (delta, delta_lo) = two_sum(d_hi, d_err + (b.mean_lo - a.mean_lo));
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * na * nb;

        // Compensated mean update preserving sub-ULP fractional bits
        let step_hi = nb * delta_n;
        let step_rem = (nb.mul_add(delta, -(step_hi * n)) + nb * delta_lo) / n;
        let (m_hi, m_err) = two_sum(a.mean, step_hi);
        let (mean, mean_lo) = two_sum(m_hi, (a.mean_lo + step_rem) + m_err);

        let m2 = a.m2 + b.m2 + term1;
        let m3 = a.m3
            + b.m3
            + term1 * delta_n * (na - nb)
            + 3.0 * delta_n * (na * b.m2 - nb * a.m2);
        let m4 = a.m4
            + b.m4
            + term1 * delta_n2 * (na * na - na * nb + nb * nb)
            + 6.0 * delta_n2 * (na * na * b.m2 + nb * nb * a.m2)
            + 4.0 * delta_n * (na * b.m3 - nb * a.m3);

        let (min, t_min) = if a.min < b.min || (a.min == b.min && a.t_min <= b.t_min) {
            (a.min, a.t_min)
        } else {
            (b.min, b.t_min)
        };
        let (max, t_max) = if a.max > b.max || (a.max == b.max && a.t_max <= b.t_max) {
            (a.max, a.t_max)
        } else {
            (b.max, b.t_max)
        };

        Self {
            n: n_u64,
            mean,
            mean_lo,
            m2,
            m3,
            m4,
            min,
            max,
            t_min,
            t_max,
            t_first: a.t_first.min(b.t_first),
            t_last: a.t_last.max(b.t_last),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_offset_no_cancellation_and_associative() {
        let base = 1e12;
        let vals = [base + 1.0, base + 2.0, base + 4.0, base + 9.0, base + 3.0];
        let elems: Vec<_> = vals
            .iter()
            .enumerate()
            .map(|(i, &v)| MomentsMonoid::singleton(i as i64, v).unwrap())
            .collect();

        let left_fold = elems
            .iter()
            .fold(MomentsMonoid::identity(&()), |a, b| MomentsMonoid::combine(&a, b, &()));
        let right_fold = elems
            .iter()
            .rev()
            .fold(MomentsMonoid::identity(&()), |b, a| MomentsMonoid::combine(a, &b, &()));

        let centered = [1.0, 2.0, 4.0, 9.0, 3.0];
        let mean_c = centered.iter().sum::<f64>() / 5.0;
        let var_c = centered.iter().map(|x| (x - mean_c).powi(2)).sum::<f64>() / 5.0;

        assert!((left_fold.variance() - var_c).abs() < 1e-10);
        assert!((right_fold.variance() - var_c).abs() < 1e-10);
        assert!((left_fold.skewness() - right_fold.skewness()).abs() < 1e-12);
        assert!((left_fold.excess_kurtosis() - right_fold.excess_kurtosis()).abs() < 1e-12);
        assert_eq!(left_fold.t_max, 3);
        assert_eq!(left_fold.t_min, 0);
    }
}
