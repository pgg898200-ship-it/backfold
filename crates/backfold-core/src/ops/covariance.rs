//! Exact O(log T) multivariate streaming covariance, correlation, and Mahalanobis anomaly monoid
//! with 106-bit Knuth TwoSum double-double mean compensation and zero-alloc inline storage (`dim <= 8`).

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_all_finite, check_finite, BackfoldError, Monoid};

const COV_INLINE_DIM: usize = 8;
const COV_INLINE_UPPER: usize = COV_INLINE_DIM * (COV_INLINE_DIM + 1) / 2; // 36

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

// Intentional: Inline stores dim<=8 buffers unboxed on stack for zero-allocation tree combines.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
enum CovStorage {
    Inline {
        mean: [f64; COV_INLINE_DIM],
        mean_lo: [f64; COV_INLINE_DIM],
        m2_upper: [f64; COV_INLINE_UPPER],
    },
    Heap {
        mean: Vec<f64>,
        mean_lo: Vec<f64>,
        m2_upper: Vec<f64>,
    },
}

#[cfg(feature = "serde")]
impl Serialize for CovStorage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            CovStorage::Inline {
                mean,
                mean_lo,
                m2_upper,
            } => (mean.as_slice(), mean_lo.as_slice(), m2_upper.as_slice()).serialize(serializer),
            CovStorage::Heap {
                mean,
                mean_lo,
                m2_upper,
            } => (mean.as_slice(), mean_lo.as_slice(), m2_upper.as_slice()).serialize(serializer),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for CovStorage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (mean, mean_lo, m2_upper): (Vec<f64>, Vec<f64>, Vec<f64>) =
            Deserialize::deserialize(deserializer)?;
        if mean.len() <= COV_INLINE_DIM
            && mean_lo.len() <= COV_INLINE_DIM
            && m2_upper.len() <= COV_INLINE_UPPER
        {
            let mut m = [0.0; COV_INLINE_DIM];
            let mut ml = [0.0; COV_INLINE_DIM];
            let mut u = [0.0; COV_INLINE_UPPER];
            m[..mean.len()].copy_from_slice(&mean);
            ml[..mean_lo.len()].copy_from_slice(&mean_lo);
            u[..m2_upper.len()].copy_from_slice(&m2_upper);
            Ok(CovStorage::Inline {
                mean: m,
                mean_lo: ml,
                m2_upper: u,
            })
        } else {
            Ok(CovStorage::Heap {
                mean,
                mean_lo,
                m2_upper,
            })
        }
    }
}

/// Associative multivariate covariance, correlation, and Mahalanobis anomaly monoid (`Ctx = usize` dim).
///
/// Uses Pébay's exact centered cross-product merge `M2_AB = M2_A + M2_B + (n_A * n_B / n) * delta * delta^T`
/// paired with a 106-bit Knuth TwoSum compensated mean vector so covariance stays accurate to `<1e-12`
/// even on `1e12 + N(0, Sigma)` streams.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CovarianceMonoid {
    pub dim: usize,
    pub n: u64,
    pub t_first: i64,
    pub t_last: i64,
    storage: CovStorage,
}

impl CovarianceMonoid {
    pub fn singleton(ts: i64, x: &[f64]) -> Result<Self, BackfoldError> {
        let dim = x.len();
        if dim == 0 {
            return Err(BackfoldError::InvalidParameter("dim must be >= 1"));
        }
        check_all_finite(x, "x")?;
        let storage = if dim <= COV_INLINE_DIM {
            let mut mean = [0.0; COV_INLINE_DIM];
            mean[..dim].copy_from_slice(x);
            CovStorage::Inline {
                mean,
                mean_lo: [0.0; COV_INLINE_DIM],
                m2_upper: [0.0; COV_INLINE_UPPER],
            }
        } else {
            CovStorage::Heap {
                mean: x.to_vec(),
                mean_lo: vec![0.0; dim],
                m2_upper: vec![0.0; dim * (dim + 1) / 2],
            }
        };
        Ok(Self {
            dim,
            n: 1,
            t_first: ts,
            t_last: ts,
            storage,
        })
    }

    #[inline]
    pub fn mean(&self) -> &[f64] {
        match &self.storage {
            CovStorage::Inline { mean, .. } => &mean[..self.dim],
            CovStorage::Heap { mean, .. } => mean.as_slice(),
        }
    }

    #[inline]
    fn mean_lo(&self) -> &[f64] {
        match &self.storage {
            CovStorage::Inline { mean_lo, .. } => &mean_lo[..self.dim],
            CovStorage::Heap { mean_lo, .. } => mean_lo.as_slice(),
        }
    }

    #[inline]
    pub fn m2_upper(&self) -> &[f64] {
        let ulen = self.dim * (self.dim + 1) / 2;
        match &self.storage {
            CovStorage::Inline { m2_upper, .. } => &m2_upper[..ulen],
            CovStorage::Heap { m2_upper, .. } => m2_upper.as_slice(),
        }
    }

    /// Dense row-major `d x d` symmetric covariance matrix (`ddof = 0` population, `ddof = 1` sample).
    pub fn covariance(&self, ddof: u64) -> Vec<f64> {
        let d = self.dim;
        let mut out = vec![0.0; d * d];
        if self.n <= ddof {
            return out;
        }
        let inv_denom = 1.0 / ((self.n - ddof) as f64);
        let u = self.m2_upper();
        let mut k = 0usize;
        for i in 0..d {
            let diag = (u[k] * inv_denom).max(0.0);
            out[i * d + i] = diag;
            k += 1;
            for j in (i + 1)..d {
                let v = u[k] * inv_denom;
                out[i * d + j] = v;
                out[j * d + i] = v;
                k += 1;
            }
        }
        out
    }

    /// Dense row-major `d x d` Pearson correlation matrix (1.0 on non-degenerate diagonal).
    pub fn correlation(&self) -> Vec<f64> {
        let d = self.dim;
        let cov = self.covariance(0);
        let mut corr = vec![0.0; d * d];
        let mut stds = vec![0.0; d];
        for i in 0..d {
            stds[i] = cov[i * d + i].max(0.0).sqrt();
        }
        for i in 0..d {
            corr[i * d + i] = if self.n > 0 { 1.0 } else { 0.0 };
            for j in (i + 1)..d {
                let denom = stds[i] * stds[j];
                let r = if denom > 1e-300 {
                    (cov[i * d + j] / denom).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                corr[i * d + j] = r;
                corr[j * d + i] = r;
            }
        }
        corr
    }

    /// Exact squared Mahalanobis distance `(x - mean)^T (Sigma_sample + ridge_eps * I)^{-1} (x - mean)`
    /// solved via Jacobi-equilibrated Cholesky. Returns an exact multivariate anomaly score.
    pub fn mahalanobis_sq(&self, x: &[f64], ridge_eps: f64) -> Result<f64, BackfoldError> {
        let d = self.dim;
        if x.len() != d {
            return Err(BackfoldError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }
        check_all_finite(x, "x")?;
        let eps = check_finite(ridge_eps, "ridge_eps")?;
        if eps <= 0.0 {
            return Err(BackfoldError::InvalidParameter("ridge_eps must be > 0"));
        }
        if self.n == 0 {
            return Ok(0.0);
        }
        let ddof = if self.n > 1 { 1 } else { 0 };
        let mut a = self.covariance(ddof);
        let mut inv_s = vec![0.0; d];
        for i in 0..d {
            a[i * d + i] += eps;
            let diag = a[i * d + i];
            if diag <= 0.0 || !diag.is_finite() {
                return Err(BackfoldError::SingularMatrix("non-positive covariance diagonal"));
            }
            inv_s[i] = 1.0 / diag.sqrt();
        }

        // Equilibrated Cholesky: D^{-1/2} A D^{-1/2} = L L^T
        let mut l = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j] * inv_s[i] * inv_s[j];
                for p in 0..j {
                    sum -= l[i * d + p] * l[j * d + p];
                }
                if i == j {
                    let piv = sum.max(1e-14);
                    l[i * d + i] = piv.sqrt();
                } else {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }

        // Compensated residual r = x - (mean + mean_lo)
        let m = self.mean();
        let ml = self.mean_lo();
        let mut z = vec![0.0; d];
        for i in 0..d {
            let (dh, de) = two_sum(x[i], -m[i]);
            let diff = dh + (de - ml[i]);
            let mut sum = diff * inv_s[i];
            for j in 0..i {
                sum -= l[i * d + j] * z[j];
            }
            z[i] = sum / l[i * d + i];
        }
        // ||L^{-1} D^{-1/2} r||^2 == r^T A^{-1} r
        let mut sq = 0.0;
        for &zi in &z {
            sq += zi * zi;
        }
        Ok(sq.max(0.0))
    }

    /// Batch squared Mahalanobis anomaly scoring for `n_rows` vectors packed row-major in `xs_flat`.
    /// Factors the Jacobi-equilibrated Cholesky decomposition once and reuses it across all rows.
    pub fn mahalanobis_sq_batch(
        &self,
        xs_flat: &[f64],
        ridge_eps: f64,
    ) -> Result<Vec<f64>, BackfoldError> {
        let d = self.dim;
        if d == 0 || !xs_flat.len().is_multiple_of(d) {
            return Err(BackfoldError::DimensionMismatch {
                expected: d,
                got: xs_flat.len(),
            });
        }
        check_all_finite(xs_flat, "xs_flat")?;
        let eps = check_finite(ridge_eps, "ridge_eps")?;
        if eps <= 0.0 {
            return Err(BackfoldError::InvalidParameter("ridge_eps must be > 0"));
        }
        let n_rows = xs_flat.len() / d;
        if self.n == 0 || n_rows == 0 {
            return Ok(vec![0.0; n_rows]);
        }
        let ddof = if self.n > 1 { 1 } else { 0 };
        let mut a = self.covariance(ddof);
        let mut inv_s = vec![0.0; d];
        for i in 0..d {
            a[i * d + i] += eps;
            let diag = a[i * d + i];
            if diag <= 0.0 || !diag.is_finite() {
                return Err(BackfoldError::SingularMatrix("non-positive covariance diagonal"));
            }
            inv_s[i] = 1.0 / diag.sqrt();
        }
        let mut l = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j] * inv_s[i] * inv_s[j];
                for p in 0..j {
                    sum -= l[i * d + p] * l[j * d + p];
                }
                if i == j {
                    l[i * d + i] = sum.max(1e-14).sqrt();
                } else {
                    l[i * d + j] = sum / l[j * d + j];
                }
            }
        }
        let m = self.mean();
        let ml = self.mean_lo();
        let mut z = vec![0.0; d];
        let mut out = Vec::with_capacity(n_rows);
        for row in xs_flat.chunks_exact(d) {
            let mut sq = 0.0;
            for i in 0..d {
                let (dh, de) = two_sum(row[i], -m[i]);
                let diff = dh + (de - ml[i]);
                let mut sum = diff * inv_s[i];
                for j in 0..i {
                    sum -= l[i * d + j] * z[j];
                }
                let zi = sum / l[i * d + i];
                z[i] = zi;
                sq += zi * zi;
            }
            out.push(sq.max(0.0));
        }
        Ok(out)
    }

    /// Exact Principal Component Analysis (PCA) of the streaming covariance matrix via pure-safe
    /// cyclic Jacobi symmetric eigendecomposition (exact to `< 1e-13`, zero external BLAS/LAPACK).
    ///
    /// Returns eigenvalues (descending), explained variance ratios, and row-major principal
    /// component eigenvectors (`k x dim`, deterministic sign with largest-magnitude entry >= 0).
    pub fn pca(&self, ddof: u64, top_k: Option<usize>) -> CovariancePca {
        let d = self.dim;
        let k = top_k.unwrap_or(d).min(d);
        if self.n <= ddof || d == 0 || k == 0 {
            return CovariancePca {
                dim: d,
                k,
                eigenvalues: vec![0.0; k],
                explained_variance_ratio: vec![0.0; k],
                components: vec![0.0; k * d],
            };
        }
        let mut a = self.covariance(ddof);
        let mut v = vec![0.0; d * d];
        for i in 0..d {
            v[i * d + i] = 1.0;
        }
        let diag_scale = (0..d)
            .map(|i| a[i * d + i].abs())
            .fold(0.0f64, f64::max)
            .max(1e-300);

        // Cyclic Jacobi sweeps with Wilkinson tangent formula
        for _sweep in 0..64 {
            let mut max_off = 0.0f64;
            for p in 0..d {
                for q in (p + 1)..d {
                    let apq = a[p * d + q];
                    let abs_apq = apq.abs();
                    if abs_apq > max_off {
                        max_off = abs_apq;
                    }
                    if abs_apq <= 1e-15 * diag_scale {
                        continue;
                    }
                    let app = a[p * d + p];
                    let aqq = a[q * d + q];
                    let tau = (aqq - app) / (2.0 * apq);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    a[p * d + p] = app - t * apq;
                    a[q * d + q] = aqq + t * apq;
                    a[p * d + q] = 0.0;
                    a[q * d + p] = 0.0;

                    for r in 0..d {
                        if r != p && r != q {
                            let arp = a[r * d + p];
                            let arq = a[r * d + q];
                            let nrp = c * arp - s * arq;
                            let nrq = s * arp + c * arq;
                            a[r * d + p] = nrp;
                            a[p * d + r] = nrp;
                            a[r * d + q] = nrq;
                            a[q * d + r] = nrq;
                        }
                        let vrp = v[r * d + p];
                        let vrq = v[r * d + q];
                        v[r * d + p] = c * vrp - s * vrq;
                        v[r * d + q] = s * vrp + c * vrq;
                    }
                }
            }
            if max_off <= 1e-14 * diag_scale {
                break;
            }
        }

        let mut order: Vec<usize> = (0..d).collect();
        order.sort_by(|&i, &j| a[j * d + j].total_cmp(&a[i * d + i]));

        let total_var: f64 = order.iter().map(|&i| a[i * d + i].max(0.0)).sum();
        let mut eigenvalues = Vec::with_capacity(k);
        let mut explained_variance_ratio = Vec::with_capacity(k);
        let mut components = Vec::with_capacity(k * d);

        for &col in order.iter().take(k) {
            let ev = a[col * d + col].max(0.0);
            eigenvalues.push(ev);
            explained_variance_ratio.push(if total_var > 0.0 { ev / total_var } else { 0.0 });
            // Canonicalize eigenvector sign (largest-magnitude entry positive)
            let mut max_idx = 0usize;
            let mut max_val = 0.0f64;
            for r in 0..d {
                let val = v[r * d + col].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = r;
                }
            }
            let sign = if v[max_idx * d + col] < 0.0 { -1.0 } else { 1.0 };
            for r in 0..d {
                components.push(v[r * d + col] * sign);
            }
        }

        CovariancePca {
            dim: d,
            k,
            eigenvalues,
            explained_variance_ratio,
            components,
        }
    }
}

/// Exact principal component decomposition of a streaming or time-range covariance matrix.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CovariancePca {
    pub dim: usize,
    pub k: usize,
    pub eigenvalues: Vec<f64>,
    pub explained_variance_ratio: Vec<f64>,
    /// Row-major `k x dim` orthonormal principal component loading vectors.
    pub components: Vec<f64>,
}

impl Monoid for CovarianceMonoid {
    type Ctx = usize;

    fn identity(dim: &Self::Ctx) -> Self {
        let d = *dim;
        let storage = if d <= COV_INLINE_DIM {
            CovStorage::Inline {
                mean: [0.0; COV_INLINE_DIM],
                mean_lo: [0.0; COV_INLINE_DIM],
                m2_upper: [0.0; COV_INLINE_UPPER],
            }
        } else {
            CovStorage::Heap {
                mean: vec![0.0; d],
                mean_lo: vec![0.0; d],
                m2_upper: vec![0.0; d * (d + 1) / 2],
            }
        };
        Self {
            dim: d,
            n: 0,
            t_first: i64::MAX,
            t_last: i64::MIN,
            storage,
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(a: &Self, b: &Self, dim: &Self::Ctx) -> Self {
        if a.n == 0 {
            return b.clone();
        }
        if b.n == 0 {
            return a.clone();
        }
        let mut out = a.clone();
        out.combine_assign(b, dim);
        out
    }

    #[inline]
    fn combine_assign(&mut self, b: &Self, dim: &Self::Ctx) {
        if b.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = b.clone();
            return;
        }
        let d = *dim;
        debug_assert_eq!(self.dim, d);
        debug_assert_eq!(b.dim, d);

        let na = self.n as f64;
        let nb = b.n as f64;
        let n_u64 = self.n + b.n;
        let n = n_u64 as f64;
        let scale = (na * nb) / n;

        match (&mut self.storage, &b.storage) {
            (
                CovStorage::Inline {
                    mean: am,
                    mean_lo: aml,
                    m2_upper: au,
                },
                CovStorage::Inline {
                    mean: bm,
                    mean_lo: bml,
                    m2_upper: bu,
                },
            ) => {
                let mut deltas = [0.0f64; COV_INLINE_DIM];
                for i in 0..d {
                    let (dh, de) = two_sum(bm[i], -am[i]);
                    let (delta, delta_lo) = two_sum(dh, de + (bml[i] - aml[i]));
                    deltas[i] = delta;
                    let step_hi = nb * (delta / n);
                    let step_rem = (nb.mul_add(delta, -(step_hi * n)) + nb * delta_lo) / n;
                    let (mh, me) = two_sum(am[i], step_hi);
                    let (new_m, new_ml) = two_sum(mh, (aml[i] + step_rem) + me);
                    am[i] = new_m;
                    aml[i] = new_ml;
                }
                let mut k = 0usize;
                for i in 0..d {
                    let sdi = scale * deltas[i];
                    for &dj in &deltas[i..d] {
                        au[k] = au[k] + bu[k] + sdi * dj;
                        k += 1;
                    }
                }
            }
            (
                CovStorage::Heap {
                    mean: am,
                    mean_lo: aml,
                    m2_upper: au,
                },
                CovStorage::Heap {
                    mean: bm,
                    mean_lo: bml,
                    m2_upper: bu,
                },
            ) => {
                let mut deltas = vec![0.0f64; d];
                for i in 0..d {
                    let (dh, de) = two_sum(bm[i], -am[i]);
                    let (delta, delta_lo) = two_sum(dh, de + (bml[i] - aml[i]));
                    deltas[i] = delta;
                    let step_hi = nb * (delta / n);
                    let step_rem = (nb.mul_add(delta, -(step_hi * n)) + nb * delta_lo) / n;
                    let (mh, me) = two_sum(am[i], step_hi);
                    let (new_m, new_ml) = two_sum(mh, (aml[i] + step_rem) + me);
                    am[i] = new_m;
                    aml[i] = new_ml;
                }
                let mut k = 0usize;
                for i in 0..d {
                    let sdi = scale * deltas[i];
                    for &dj in &deltas[i..d] {
                        au[k] = au[k] + bu[k] + sdi * dj;
                        k += 1;
                    }
                }
            }
            _ => unreachable!("mismatched CovarianceMonoid storage variants for identical dim"),
        }

        self.n = n_u64;
        self.t_first = self.t_first.min(b.t_first);
        self.t_last = self.t_last.max(b.t_last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventKey, FoldTree};

    #[test]
    fn high_offset_covariance_and_mahalanobis_exact() {
        let base = 1e12_f64;
        let pts = [
            [1.0, 2.0, -1.0],
            [2.0, 3.5, 0.5],
            [4.0, 7.0, -2.0],
            [9.0, 14.0, 1.0],
            [3.0, 5.0, 0.0],
        ];
        let mut tree: FoldTree<CovarianceMonoid, 4> = FoldTree::new(3);
        let mut exact_pts = Vec::new();
        for (i, p) in pts.iter().enumerate() {
            let shifted = [base + p[0], base + p[1], base + p[2]];
            exact_pts.push([shifted[0] - base, shifted[1] - base, shifted[2] - base]);
            tree.upsert(
                EventKey::at(i as i64),
                CovarianceMonoid::singleton(i as i64, &shifted).unwrap(),
            );
        }
        let s = tree.summary();
        let cov = s.covariance(1);
        let mut ref_mean = [0.0; 3];
        for p in &exact_pts {
            for j in 0..3 {
                ref_mean[j] += p[j] / 5.0;
            }
        }
        let mut ref_cov = [0.0; 9];
        for p in &exact_pts {
            for i in 0..3 {
                for j in 0..3 {
                    ref_cov[i * 3 + j] += (p[i] - ref_mean[i]) * (p[j] - ref_mean[j]) / 4.0;
                }
            }
        }
        for i in 0..9 {
            assert!((cov[i] - ref_cov[i]).abs() < 1e-11, "cov[{i}] diff");
        }
        let corr = s.correlation();
        assert!((corr[0] - 1.0).abs() < 1e-12);
        assert!(corr[1] > 0.99);

        // Note: 1e12 has ULP = 2^-13 (~1.22e-4), so base + 3.8 rounds in f64 whereas our
        // 106-bit TwoSum mean stores 1e12 + 3.8 exactly. Test with an exact-in-f64 query
        // and verify against the exact shifted-origin Mahalanobis distance.
        let q_exact = [base + 4.0, base + 6.5, base - 0.5];
        let d2_near = s.mahalanobis_sq(&q_exact, 1e-6).unwrap();
        let d2_out = s
            .mahalanobis_sq(&[base + 50.0, base - 50.0, base + 20.0], 1e-6)
            .unwrap();
        assert!(d2_near < 5.0, "near point should have small d2, got {d2_near}");
        assert!(d2_out > 100.0);

        // Centered copy (no 1e12 offset) must produce identical Mahalanobis distance to <1e-11!
        let mut centered_tree: FoldTree<CovarianceMonoid, 4> = FoldTree::new(3);
        for (i, p) in exact_pts.iter().enumerate() {
            centered_tree.upsert(
                EventKey::at(i as i64),
                CovarianceMonoid::singleton(i as i64, p).unwrap(),
            );
        }
        let sc = centered_tree.summary();
        let d2_near_ref = sc
            .mahalanobis_sq(&[q_exact[0] - base, q_exact[1] - base, q_exact[2] - base], 1e-6)
            .unwrap();
        let d2_out_ref = sc.mahalanobis_sq(&[50.0, -50.0, 20.0], 1e-6).unwrap();
        assert!(
            (d2_near - d2_near_ref).abs() < 1e-11,
            "1e12 offset d2_near={d2_near} vs ref={d2_near_ref}"
        );
        assert!(
            (d2_out - d2_out_ref).abs() < 1e-8,
            "1e12 offset d2_out={d2_out} vs ref={d2_out_ref}"
        );
        let d2_zero = sc.mahalanobis_sq(&ref_mean, 1e-6).unwrap();
        assert!(d2_zero < 1e-14, "exact center d2={d2_zero}");
    }

    #[test]
    fn streaming_pca_orthonormal_reconstruction_and_1e12_parity() {
        let base = 1e12_f64;
        let mut tree_hi: FoldTree<CovarianceMonoid, 4> = FoldTree::new(4);
        let mut tree_lo: FoldTree<CovarianceMonoid, 4> = FoldTree::new(4);
        let mut seed = 123456789u64;
        for i in 0..400 {
            let mut next_u = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
            };
            let z0 = next_u() * 10.0;
            let z1 = next_u() * 3.0;
            let z2 = next_u() * 0.5;
            let z3 = next_u() * 0.1;
            let raw = [
                z0 + 0.2 * z1,
                0.8 * z0 - 0.6 * z1 + z2,
                -0.4 * z0 + z1 + 0.3 * z3,
                0.1 * z0 - 0.3 * z1 + z3,
            ];
            let shifted = [base + raw[0], base + raw[1], base + raw[2], base + raw[3]];
            let exact = [
                shifted[0] - base,
                shifted[1] - base,
                shifted[2] - base,
                shifted[3] - base,
            ];
            tree_hi.upsert(
                EventKey::at(i),
                CovarianceMonoid::singleton(i, &shifted).unwrap(),
            );
            tree_lo.upsert(
                EventKey::at(i),
                CovarianceMonoid::singleton(i, &exact).unwrap(),
            );
        }
        let s_hi = tree_hi.summary();
        let s_lo = tree_lo.summary();
        let pca_hi = s_hi.pca(1, None);
        let pca_lo = s_lo.pca(1, None);
        let cov_lo = s_lo.covariance(1);
        assert_eq!(pca_hi.eigenvalues.len(), 4);
        // Eigenvalues strictly descending and explained variance sums to 1
        let sum_evr: f64 = pca_hi.explained_variance_ratio.iter().sum();
        assert!((sum_evr - 1.0).abs() < 1e-13);
        for i in 0..3 {
            assert!(pca_hi.eigenvalues[i] >= pca_hi.eigenvalues[i + 1]);
        }
        // Orthonormality V * V^T == I and exact reconstruction V^T * diag(lambda) * V == Cov
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0;
                let mut rec = 0.0;
                for r in 0..4 {
                    dot += pca_hi.components[i * 4 + r] * pca_hi.components[j * 4 + r];
                    rec += pca_hi.eigenvalues[r]
                        * pca_hi.components[r * 4 + i]
                        * pca_hi.components[r * 4 + j];
                }
                let expected_i = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected_i).abs() < 1e-13, "V*V^T[{i},{j}]={dot}");
                assert!(
                    (rec - cov_lo[i * 4 + j]).abs() < 1e-11,
                    "reconstruct[{i},{j}]: {rec} vs {}",
                    cov_lo[i * 4 + j]
                );
            }
            assert!(
                (pca_hi.eigenvalues[i] - pca_lo.eigenvalues[i]).abs() < 1e-11,
                "1e12 offset eigenvalue[{i}] mismatch"
            );
        }
    }
}
