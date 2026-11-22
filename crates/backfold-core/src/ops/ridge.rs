//! Continuous-time exponential-forgetting online Ridge regression associative monoid
//! with Jacobi-equilibrated Cholesky solver and exact predictive variance.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_all_finite, check_finite, BackfoldError, Monoid};

/// Solved Ridge regression weights and diagnostic statistics.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RidgeSolution {
    pub weights: Vec<f64>,
    pub residual_variance: f64,
    pub effective_weight: f64,
    pub n: u64,
    pub condition_est: f64,
}

impl RidgeSolution {
    pub fn predict(&self, x: &[f64]) -> Result<f64, BackfoldError> {
        if x.len() != self.weights.len() {
            return Err(BackfoldError::DimensionMismatch {
                expected: self.weights.len(),
                got: x.len(),
            });
        }
        check_all_finite(x, "x")?;
        let mut s = 0.0;
        for (&w, &xi) in self.weights.iter().zip(x.iter()) {
            s += w * xi;
        }
        Ok(s)
    }
}

const RIDGE_INLINE_DIM: usize = 8;
const RIDGE_INLINE_GRAM: usize = RIDGE_INLINE_DIM * (RIDGE_INLINE_DIM + 1) / 2; // 36

// Intentional: Inline holds small-dim (<=8) buffers unboxed on stack for zero-allocation combine.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
enum RidgeStorage {
    Inline {
        gram: [f64; RIDGE_INLINE_GRAM],
        gy: [f64; RIDGE_INLINE_DIM],
    },
    Heap {
        gram: Vec<f64>,
        gy: Vec<f64>,
    },
}

#[cfg(feature = "serde")]
impl Serialize for RidgeStorage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            RidgeStorage::Inline { gram, gy } => (gram.as_slice(), gy.as_slice()).serialize(serializer),
            RidgeStorage::Heap { gram, gy } => (gram.as_slice(), gy.as_slice()).serialize(serializer),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for RidgeStorage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (gram, gy): (Vec<f64>, Vec<f64>) = Deserialize::deserialize(deserializer)?;
        if gy.len() <= RIDGE_INLINE_DIM && gram.len() <= RIDGE_INLINE_GRAM {
            let mut g_arr = [0.0; RIDGE_INLINE_GRAM];
            let mut y_arr = [0.0; RIDGE_INLINE_DIM];
            g_arr[..gram.len()].copy_from_slice(&gram);
            y_arr[..gy.len()].copy_from_slice(&gy);
            Ok(RidgeStorage::Inline {
                gram: g_arr,
                gy: y_arr,
            })
        } else {
            Ok(RidgeStorage::Heap { gram, gy })
        }
    }
}

/// Associative sufficient-statistics monoid for continuous-time discounted Ridge / RLS regression.
///
/// Context `Ctx = (usize, f64)` holds `(feature_dim, decay_rate)` where `decay_rate >= 0.0`.
/// Stores packed upper-triangular Gram matrix `G = X^T W X` (`d*(d+1)/2`), `g_y = X^T W y`,
/// and `y^T W y`, all discounted to `t_last` via `w_i = weight_i * exp(-decay_rate * (t_last - t_i))`.
/// Uses zero-heap-allocation inline stack arrays for `dim <= 8`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RidgeMonoid {
    pub dim: usize,
    pub n: u64,
    pub t_first: i64,
    pub t_last: i64,
    pub w_sum: f64,
    storage: RidgeStorage,
    pub yy: f64,
}

#[inline]
fn upper_idx(dim: usize, i: usize, j: usize) -> usize {
    debug_assert!(i <= j && j < dim);
    i * dim - (i * (i.saturating_sub(1))) / 2 + (j - i)
}

impl RidgeMonoid {
    #[inline]
    pub fn gram_upper(&self) -> &[f64] {
        let glen = self.dim * (self.dim + 1) / 2;
        match &self.storage {
            RidgeStorage::Inline { gram, .. } => &gram[..glen],
            RidgeStorage::Heap { gram, .. } => gram.as_slice(),
        }
    }

    #[inline]
    pub fn gy(&self) -> &[f64] {
        match &self.storage {
            RidgeStorage::Inline { gy, .. } => &gy[..self.dim],
            RidgeStorage::Heap { gy, .. } => gy.as_slice(),
        }
    }

    pub fn singleton(ts: i64, x: &[f64], y: f64, weight: f64) -> Result<Self, BackfoldError> {
        let dim = x.len();
        if dim == 0 {
            return Err(BackfoldError::InvalidParameter("feature dim must be >= 1"));
        }
        check_all_finite(x, "x")?;
        let y = check_finite(y, "y")?;
        let weight = check_finite(weight, "weight")?;
        if weight <= 0.0 {
            return Err(BackfoldError::InvalidParameter("weight must be > 0"));
        }
        let storage = if dim <= RIDGE_INLINE_DIM {
            let mut gram = [0.0f64; RIDGE_INLINE_GRAM];
            let mut gy = [0.0f64; RIDGE_INLINE_DIM];
            let mut k = 0usize;
            for (i, &xi) in x.iter().enumerate() {
                let wxi = weight * xi;
                gy[i] = wxi * y;
                for &xj in &x[i..] {
                    gram[k] = wxi * xj;
                    k += 1;
                }
            }
            RidgeStorage::Inline { gram, gy }
        } else {
            let mut gram = Vec::with_capacity(dim * (dim + 1) / 2);
            let mut gy = Vec::with_capacity(dim);
            for (i, &xi) in x.iter().enumerate() {
                let wxi = weight * xi;
                gy.push(wxi * y);
                for &xj in &x[i..] {
                    gram.push(wxi * xj);
                }
            }
            RidgeStorage::Heap { gram, gy }
        };
        Ok(Self {
            dim,
            n: 1,
            t_first: ts,
            t_last: ts,
            w_sum: weight,
            storage,
            yy: weight * y * y,
        })
    }

    /// Build dense symmetric `A = G + alpha * I`.
    fn build_dense_system(&self, alpha: f64) -> Vec<f64> {
        let d = self.dim;
        let gram = self.gram_upper();
        let mut a = vec![0.0; d * d];
        let mut k = 0usize;
        for i in 0..d {
            for j in i..d {
                let v = gram[k];
                k += 1;
                a[i * d + j] = v;
                a[j * d + i] = v;
            }
            a[i * d + i] += alpha;
        }
        a
    }

    /// Jacobi-equilibrated Cholesky factorization `D^{-1/2} (G + alpha I) D^{-1/2} + rel_jitter * I = L L^T`.
    fn factor_equilibrated_with_jitter(
        &self,
        alpha: f64,
        rel_jitter: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), BackfoldError> {
        let d = self.dim;
        let a = self.build_dense_system(alpha);
        let mut inv_sqrt_diag = vec![0.0; d];
        let mut d_min = f64::INFINITY;
        let mut d_max = 0.0f64;
        for i in 0..d {
            let diag = a[i * d + i];
            if diag <= 0.0 || !diag.is_finite() {
                return Err(BackfoldError::SingularMatrix("non-positive diagonal in G + alpha I"));
            }
            d_min = d_min.min(diag);
            d_max = d_max.max(diag);
            inv_sqrt_diag[i] = 1.0 / diag.sqrt();
        }
        let cond_est = d_max / d_min.max(1e-300);

        let mut l = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j] * inv_sqrt_diag[i] * inv_sqrt_diag[j];
                if i == j {
                    sum += rel_jitter;
                }
                for p in 0..j {
                    sum -= l[i * d + p] * l[j * d + p];
                }
                if i == j {
                    if sum <= 1e-15 || !sum.is_finite() {
                        return Err(BackfoldError::SingularMatrix(
                            "Cholesky pivot failed after Jacobi equilibration",
                        ));
                    }
                    l[i * d + i] = sum.sqrt();
                } else {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }
        Ok((l, inv_sqrt_diag, cond_est))
    }

    fn factor_equilibrated(&self, alpha: f64) -> Result<(Vec<f64>, Vec<f64>, f64), BackfoldError> {
        self.factor_equilibrated_with_jitter(alpha, 0.0)
    }

    fn solve_cholesky_equilibrated(
        d: usize,
        l: &[f64],
        inv_sqrt_diag: &[f64],
        rhs: &[f64],
    ) -> Vec<f64> {
        let mut z = vec![0.0; d];
        for i in 0..d {
            let mut sum = rhs[i] * inv_sqrt_diag[i];
            for p in 0..i {
                sum -= l[i * d + p] * z[p];
            }
            z[i] = sum / l[i * d + i];
        }
        let mut w = vec![0.0; d];
        for i in (0..d).rev() {
            let mut sum = z[i];
            for p in (i + 1)..d {
                sum -= l[p * d + i] * w[p];
            }
            w[i] = sum / l[i * d + i];
        }
        for i in 0..d {
            w[i] *= inv_sqrt_diag[i];
        }
        w
    }

    /// Fallback partial-pivoted Gaussian elimination on Jacobi-equilibrated system.
    fn solve_pivoted_fallback(&self, alpha: f64, rhs: &[f64]) -> Result<Vec<f64>, BackfoldError> {
        let d = self.dim;
        let a_raw = self.build_dense_system(alpha);
        let mut inv_s = vec![1.0; d];
        for i in 0..d {
            let diag = a_raw[i * d + i].abs();
            if diag > 1e-30 {
                inv_s[i] = 1.0 / diag.sqrt();
            }
        }
        let mut a = vec![0.0; d * d];
        let mut b = vec![0.0; d];
        for i in 0..d {
            b[i] = rhs[i] * inv_s[i];
            for j in 0..d {
                a[i * d + j] = a_raw[i * d + j] * inv_s[i] * inv_s[j];
            }
        }
        for col in 0..d {
            let mut piv = col;
            let mut best = a[col * d + col].abs();
            for r in (col + 1)..d {
                let v = a[r * d + col].abs();
                if v > best {
                    best = v;
                    piv = r;
                }
            }
            if best <= 1e-18 || !best.is_finite() {
                return Err(BackfoldError::SingularMatrix("singular Ridge normal equations"));
            }
            if piv != col {
                for j in col..d {
                    a.swap(col * d + j, piv * d + j);
                }
                b.swap(col, piv);
            }
            let pivot_val = a[col * d + col];
            for r in (col + 1)..d {
                let factor = a[r * d + col] / pivot_val;
                b[r] -= factor * b[col];
                for j in col..d {
                    a[r * d + j] -= factor * a[col * d + j];
                }
            }
        }
        let mut x = vec![0.0; d];
        for i in (0..d).rev() {
            let mut sum = b[i];
            for j in (i + 1)..d {
                sum -= a[i * d + j] * x[j];
            }
            x[i] = sum / a[i * d + i];
        }
        for i in 0..d {
            x[i] *= inv_s[i];
        }
        Ok(x)
    }

    /// Solve `(G + alpha I) w = g_y` via Jacobi-equilibrated Cholesky (with pivoted LU fallback).
    pub fn solve(&self, alpha: f64) -> Result<RidgeSolution, BackfoldError> {
        let alpha = check_finite(alpha, "alpha")?;
        if alpha < 0.0 {
            return Err(BackfoldError::InvalidParameter("alpha must be >= 0"));
        }
        let d = self.dim;
        if self.n == 0 {
            return Ok(RidgeSolution {
                weights: vec![0.0; d],
                residual_variance: 0.0,
                effective_weight: 0.0,
                n: 0,
                condition_est: 1.0,
            });
        }

        let gy = self.gy();
        let gram = self.gram_upper();
        let (weights, cond_est) = match self.factor_equilibrated(alpha) {
            Ok((l, inv_s, cond)) => (
                Self::solve_cholesky_equilibrated(d, &l, &inv_s, gy),
                cond,
            ),
            Err(_) => match self.solve_pivoted_fallback(alpha, gy) {
                Ok(w) => (w, f64::INFINITY),
                Err(err) if alpha > 0.0 => {
                    let (l, inv_s, cond) = self
                        .factor_equilibrated_with_jitter(alpha, 1e-12)
                        .map_err(|_| err)?;
                    (Self::solve_cholesky_equilibrated(d, &l, &inv_s, gy), cond)
                }
                Err(err) => return Err(err),
            },
        };

        // Exact non-negative RSS = y^T W y - 2 w^T g_y + w^T G w
        let mut w_gy = 0.0;
        let mut w_g_w = 0.0;
        for i in 0..d {
            w_gy += weights[i] * gy[i];
            let diag = gram[upper_idx(d, i, i)];
            w_g_w += weights[i] * weights[i] * diag;
            for j in (i + 1)..d {
                let gij = gram[upper_idx(d, i, j)];
                w_g_w += 2.0 * weights[i] * weights[j] * gij;
            }
        }
        let rss = (self.yy - 2.0 * w_gy + w_g_w).max(0.0);
        let denom = (self.w_sum - d as f64).max(1.0);
        Ok(RidgeSolution {
            weights,
            residual_variance: rss / denom,
            effective_weight: self.w_sum,
            n: self.n,
            condition_est: cond_est,
        })
    }

    /// Predict `(y_hat, pred_var)` where `pred_var = sigma2 * (1 + x^T (G + alpha I)^{-1} x)`.
    pub fn predict_with_variance(
        &self,
        x: &[f64],
        alpha: f64,
    ) -> Result<(f64, f64), BackfoldError> {
        if x.len() != self.dim {
            return Err(BackfoldError::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        check_all_finite(x, "x")?;
        let sol = self.solve(alpha)?;
        let y_hat = sol.predict(x)?;
        if self.n == 0 {
            return Ok((0.0, 0.0));
        }
        let d = self.dim;
        let inv_a_x = match self.factor_equilibrated(alpha) {
            Ok((l, inv_s, _)) => Self::solve_cholesky_equilibrated(d, &l, &inv_s, x),
            Err(_) => match self.solve_pivoted_fallback(alpha, x) {
                Ok(v) => v,
                Err(err) if alpha > 0.0 => {
                    let (l, inv_s, _) = self
                        .factor_equilibrated_with_jitter(alpha, 1e-12)
                        .map_err(|_| err)?;
                    Self::solve_cholesky_equilibrated(d, &l, &inv_s, x)
                }
                Err(err) => return Err(err),
            },
        };
        let mut leverage = 0.0;
        for i in 0..d {
            leverage += x[i] * inv_a_x[i];
        }
        let pred_var = sol.residual_variance * (1.0 + leverage.max(0.0));
        Ok((y_hat, pred_var))
    }
}

impl Monoid for RidgeMonoid {
    /// `(feature_dim, continuous_time_decay_rate)`
    type Ctx = (usize, f64);

    fn identity(ctx: &Self::Ctx) -> Self {
        let d = ctx.0;
        let storage = if d <= RIDGE_INLINE_DIM {
            RidgeStorage::Inline {
                gram: [0.0; RIDGE_INLINE_GRAM],
                gy: [0.0; RIDGE_INLINE_DIM],
            }
        } else {
            RidgeStorage::Heap {
                gram: vec![0.0; d * (d + 1) / 2],
                gy: vec![0.0; d],
            }
        };
        Self {
            dim: d,
            n: 0,
            t_first: i64::MAX,
            t_last: i64::MIN,
            w_sum: 0.0,
            storage,
            yy: 0.0,
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(left: &Self, right: &Self, ctx: &Self::Ctx) -> Self {
        if left.n == 0 {
            return right.clone();
        }
        if right.n == 0 {
            return left.clone();
        }
        let d = ctx.0;
        let lambda = ctx.1.max(0.0);
        debug_assert_eq!(left.dim, d);
        debug_assert_eq!(right.dim, d);

        let dt = right.t_last.saturating_sub(left.t_last).max(0) as f64;
        let arg = lambda * dt;
        let decay = if arg > 680.0 { 0.0 } else { (-arg).exp() };

        let storage = match (&left.storage, &right.storage) {
            (
                RidgeStorage::Inline { gram: lg, gy: ly },
                RidgeStorage::Inline { gram: rg, gy: ry },
            ) => {
                let glen = d * (d + 1) / 2;
                let mut gram = [0.0f64; RIDGE_INLINE_GRAM];
                let mut gy = [0.0f64; RIDGE_INLINE_DIM];
                if decay == 0.0 {
                    gram[..glen].copy_from_slice(&rg[..glen]);
                    gy[..d].copy_from_slice(&ry[..d]);
                } else {
                    for i in 0..glen {
                        gram[i] = decay.mul_add(lg[i], rg[i]);
                    }
                    for i in 0..d {
                        gy[i] = decay.mul_add(ly[i], ry[i]);
                    }
                }
                RidgeStorage::Inline { gram, gy }
            }
            _ => {
                let lg = left.gram_upper();
                let rg = right.gram_upper();
                let ly = left.gy();
                let ry = right.gy();
                let mut gram = Vec::with_capacity(lg.len());
                for (&a, &b) in lg.iter().zip(rg.iter()) {
                    gram.push(decay.mul_add(a, b));
                }
                let mut gy = Vec::with_capacity(d);
                for (&a, &b) in ly.iter().zip(ry.iter()) {
                    gy.push(decay.mul_add(a, b));
                }
                RidgeStorage::Heap { gram, gy }
            }
        };
        Self {
            dim: d,
            n: left.n + right.n,
            t_first: left.t_first.min(right.t_first),
            t_last: right.t_last,
            w_sum: decay.mul_add(left.w_sum, right.w_sum),
            storage,
            yy: decay.mul_add(left.yy, right.yy),
        }
    }

    #[inline]
    fn combine_assign(&mut self, right: &Self, ctx: &Self::Ctx) {
        if right.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = right.clone();
            return;
        }
        let d = ctx.0;
        let lambda = ctx.1.max(0.0);
        let dt = right.t_last.saturating_sub(self.t_last).max(0) as f64;
        let arg = lambda * dt;
        let decay = if arg > 680.0 { 0.0 } else { (-arg).exp() };

        match (&mut self.storage, &right.storage) {
            (
                RidgeStorage::Inline { gram: lg, gy: ly },
                RidgeStorage::Inline { gram: rg, gy: ry },
            ) => {
                let glen = d * (d + 1) / 2;
                if decay == 0.0 {
                    lg[..glen].copy_from_slice(&rg[..glen]);
                    ly[..d].copy_from_slice(&ry[..d]);
                } else {
                    for i in 0..glen {
                        lg[i] = decay.mul_add(lg[i], rg[i]);
                    }
                    for i in 0..d {
                        ly[i] = decay.mul_add(ly[i], ry[i]);
                    }
                }
            }
            (
                RidgeStorage::Heap { gram: lg, gy: ly },
                RidgeStorage::Heap { gram: rg, gy: ry },
            ) => {
                for (a, &b) in lg.iter_mut().zip(rg.iter()) {
                    *a = decay.mul_add(*a, b);
                }
                for (a, &b) in ly.iter_mut().zip(ry.iter()) {
                    *a = decay.mul_add(*a, b);
                }
            }
            _ => {
                *self = Self::combine(self, right, ctx);
                return;
            }
        }
        self.n += right.n;
        self.t_first = self.t_first.min(right.t_first);
        self.t_last = right.t_last;
        self.w_sum = decay.mul_add(self.w_sum, right.w_sum);
        self.yy = decay.mul_add(self.yy, right.yy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ill_conditioned_jacobi_ridge_matches_truth() {
        // Feature 0 has scale 1e5, feature 1 has scale 1e-5 -> Gram cond ~ 1e20 without equilibration!
        let ctx = (2usize, 0.0f64);
        let mut acc = RidgeMonoid::identity(&ctx);
        for i in 1..=200i64 {
            let f = i as f64;
            let x = [1e5 * (f * 0.1).sin(), 1e-5 * (f * 0.37).cos()];
            let y = 2e-5 * x[0] - 3e5 * x[1];
            acc = RidgeMonoid::combine(&acc, &RidgeMonoid::singleton(i, &x, y, 1.0).unwrap(), &ctx);
        }
        let sol = acc.solve(1e-18).unwrap();
        assert!((sol.weights[0] - 2e-5).abs() < 1e-10);
        assert!((sol.weights[1] + 3e5).abs() < 1e-2);
    }
}
