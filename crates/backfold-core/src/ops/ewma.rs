//! Continuous-time irregular-dt EWMA and discrete-step affine EWMA associative monoids.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_finite, BackfoldError, Monoid};

/// Continuous-time exponentially weighted centered mean and variance over arbitrary timestamps.
///
/// Context `Ctx = f64` is the non-negative decay rate `lambda = ln(2) / half_life`.
/// For an interval ending at `t_last`, each point `(t_i, x_i)` carries weight
/// `w_i = exp(-lambda * (t_last - t_i))`. Because centered weighted moments depend solely
/// on the weighted multiset of observations, composition is exact and associative.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EwmaMonoid {
    pub n: u64,
    pub w_sum: f64,
    pub mean: f64,
    pub m2: f64,
    pub t_first: i64,
    pub t_last: i64,
}

impl EwmaMonoid {
    pub fn decay_rate_from_half_life(half_life: f64) -> Result<f64, BackfoldError> {
        let hl = check_finite(half_life, "half_life")?;
        if hl <= 0.0 {
            return Err(BackfoldError::InvalidParameter("half_life must be > 0"));
        }
        Ok(std::f64::consts::LN_2 / hl)
    }

    pub fn singleton(ts: i64, x: f64) -> Result<Self, BackfoldError> {
        let x = check_finite(x, "x")?;
        Ok(Self {
            n: 1,
            w_sum: 1.0,
            mean: x,
            m2: 0.0,
            t_first: ts,
            t_last: ts,
        })
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        if self.w_sum <= 0.0 {
            0.0
        } else {
            (self.m2 / self.w_sum).max(0.0)
        }
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Monoid for EwmaMonoid {
    type Ctx = f64;

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            n: 0,
            w_sum: 0.0,
            mean: 0.0,
            m2: 0.0,
            t_first: i64::MAX,
            t_last: i64::MIN,
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(left: &Self, right: &Self, lambda: &Self::Ctx) -> Self {
        if left.n == 0 {
            return right.clone();
        }
        if right.n == 0 {
            return left.clone();
        }
        let dt = right.t_last.saturating_sub(left.t_last).max(0) as f64;
        let arg = lambda.max(0.0) * dt;
        // Subnormal-safe clamp: exp(-700) ~ 1e-304
        let decay = if arg > 700.0 { 0.0 } else { (-arg).exp() };
        let wa = decay * left.w_sum;
        let wb = right.w_sum;
        let w = wa + wb;
        if wa <= 0.0 {
            return Self {
                n: left.n + right.n,
                w_sum: wb,
                mean: right.mean,
                m2: right.m2,
                t_first: left.t_first.min(right.t_first),
                t_last: right.t_last,
            };
        }
        let delta = right.mean - left.mean;
        let mean = left.mean + delta * (wb / w);
        let m2 = decay * left.m2 + right.m2 + delta * delta * (wa * wb / w);
        Self {
            n: left.n + right.n,
            w_sum: w,
            mean,
            m2,
            t_first: left.t_first.min(right.t_first),
            t_last: right.t_last,
        }
    }
}

/// Exact discrete-step EWMA `s_t = (1 - alpha) * s_{t-1} + alpha * x_t` via affine composition.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AffineEwmaMonoid {
    pub a: f64,
    pub b: f64,
    pub n: u64,
}

impl AffineEwmaMonoid {
    pub fn singleton(x: f64, alpha: f64) -> Result<Self, BackfoldError> {
        let x = check_finite(x, "x")?;
        let alpha = check_finite(alpha, "alpha")?;
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(BackfoldError::InvalidParameter("alpha must be in (0, 1]"));
        }
        Ok(Self {
            a: 1.0 - alpha,
            b: alpha * x,
            n: 1,
        })
    }

    #[inline]
    pub fn apply(&self, s0: f64) -> f64 {
        self.a * s0 + self.b
    }
}

impl Monoid for AffineEwmaMonoid {
    type Ctx = ();

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            n: 0,
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
            a: right.a * left.a,
            b: right.a * left.b + right.b,
            n: left.n + right.n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_ewma_matches_direct_formula_irregular_dt() {
        let ts = [0i64, 3, 4, 15, 42, 43, 100];
        let xs = [1.2, -0.5, 3.4, 2.1, -1.7, 0.8, 4.5];
        let lam = EwmaMonoid::decay_rate_from_half_life(20.0).unwrap();

        let elems: Vec<_> = ts
            .iter()
            .zip(xs.iter())
            .map(|(&t, &x)| EwmaMonoid::singleton(t, x).unwrap())
            .collect();
        let left = elems
            .iter()
            .fold(EwmaMonoid::identity(&lam), |a, b| EwmaMonoid::combine(&a, b, &lam));
        let right = elems
            .iter()
            .rev()
            .fold(EwmaMonoid::identity(&lam), |b, a| EwmaMonoid::combine(a, &b, &lam));

        let t_last = *ts.last().unwrap() as f64;
        let ws: Vec<f64> = ts.iter().map(|&t| (-lam * (t_last - t as f64)).exp()).collect();
        let w_sum: f64 = ws.iter().sum();
        let mean: f64 = ws.iter().zip(xs.iter()).map(|(w, x)| w * x).sum::<f64>() / w_sum;
        let var: f64 = ws
            .iter()
            .zip(xs.iter())
            .map(|(w, x)| w * (x - mean).powi(2))
            .sum::<f64>()
            / w_sum;

        assert!((left.mean - mean).abs() < 1e-13);
        assert!((left.variance() - var).abs() < 1e-13);
        assert!((left.mean - right.mean).abs() < 1e-13);
        assert!((left.variance() - right.variance()).abs() < 1e-13);
    }

    #[test]
    fn affine_ewma_exact_match() {
        let xs = [1.0, 4.0, -2.0, 5.0, 3.0];
        let alpha = 0.25;
        let mut s = 0.5;
        let mut acc = AffineEwmaMonoid::identity(&());
        for &x in &xs {
            s = (1.0 - alpha) * s + alpha * x;
            acc = AffineEwmaMonoid::combine(&acc, &AffineEwmaMonoid::singleton(x, alpha).unwrap(), &());
        }
        assert!((acc.apply(0.5) - s).abs() < 1e-14);
    }
}
