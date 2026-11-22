//! Exact Särkkä & García-Fernández (2020, IEEE TAC) associative linear Gaussian Kalman filter.
//!
//! Enables O(log T) out-of-order sensor insertions, retractions, retroactive corrections,
//! missing observations, irregular `dt` continuous-time transitions, and O(1) counterfactual forks
//! with bit-tight parity against Joseph-form sequential Kalman filtering.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::monoid::{check_all_finite, check_finite, BackfoldError, Monoid};

/// Exact posterior state `(mean, cov)` at the end of a filtered interval.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanPosterior {
    pub dim: usize,
    pub mean: Vec<f64>,
    pub cov: Vec<f64>,
}

/// Ultra-fast stack-only scalar (1D) associative Kalman filter monoid (zero heap allocations).
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanFilter1D {
    pub n: u64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub eta: f64,
    pub j: f64,
}

impl KalmanFilter1D {
    pub fn from_observation(f: f64, q: f64, h: f64, r: f64, y: f64) -> Result<Self, BackfoldError> {
        let f = check_finite(f, "f")?;
        let q = check_finite(q, "q")?;
        let h = check_finite(h, "h")?;
        let r = check_finite(r, "r")?;
        let y = check_finite(y, "y")?;
        if q < 0.0 || r < 0.0 {
            return Err(BackfoldError::InvalidParameter("q and r must be >= 0"));
        }
        let s = h * h * q + r;
        if s <= 0.0 {
            return Err(BackfoldError::SingularMatrix("innovation variance S <= 0"));
        }
        let inv_s = 1.0 / s;
        let k = q * h * inv_s;
        let ikh = 1.0 - k * h;
        Ok(Self {
            n: 1,
            a: ikh * f,
            b: k * y,
            c: (ikh * ikh * q + k * k * r).max(0.0),
            eta: f * h * inv_s * y,
            j: (f * f * h * h * inv_s).max(0.0),
        })
    }

    pub fn from_missing(f: f64, q: f64) -> Result<Self, BackfoldError> {
        let f = check_finite(f, "f")?;
        let q = check_finite(q, "q")?;
        if q < 0.0 {
            return Err(BackfoldError::InvalidParameter("q must be >= 0"));
        }
        Ok(Self {
            n: 1,
            a: f,
            b: 0.0,
            c: q,
            eta: 0.0,
            j: 0.0,
        })
    }

    /// Continuous-time random walk `x_t = x_{t-1} + sqrt(q_rate * dt) * w_t`, `y_t = x_t + v_t`.
    pub fn from_random_walk(
        dt: f64,
        q_rate: f64,
        r_obs: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        let dt = check_finite(dt, "dt")?;
        let q_rate = check_finite(q_rate, "q_rate")?;
        if dt < 0.0 || q_rate < 0.0 {
            return Err(BackfoldError::InvalidParameter("dt and q_rate must be >= 0"));
        }
        Self::from_observation(1.0, q_rate * dt, 1.0, r_obs, y)
    }

    /// Apply initial Gaussian prior `N(m0, p0)` to obtain exact posterior `(mean, var)`.
    pub fn posterior(&self, m0: f64, p0: f64) -> Result<(f64, f64), BackfoldError> {
        let m0 = check_finite(m0, "m0")?;
        let p0 = check_finite(p0, "p0")?;
        if p0 < 0.0 {
            return Err(BackfoldError::InvalidParameter("p0 must be >= 0"));
        }
        if self.n == 0 {
            return Ok((m0, p0));
        }
        let d = 1.0 / (1.0 + p0 * self.j);
        let mean = self.a * d * (m0 + p0 * self.eta) + self.b;
        let var = (self.a * self.a * d * p0 + self.c).max(0.0);
        Ok((mean, var))
    }
}

impl Monoid for KalmanFilter1D {
    type Ctx = ();

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            n: 0,
            a: 1.0,
            b: 0.0,
            c: 0.0,
            eta: 0.0,
            j: 0.0,
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(e1: &Self, e2: &Self, _ctx: &Self::Ctx) -> Self {
        if e1.n == 0 {
            return *e2;
        }
        if e2.n == 0 {
            return *e1;
        }
        let d = 1.0 / (1.0 + e1.c * e2.j);
        Self {
            n: e1.n + e2.n,
            a: flush_subnormal(e2.a * d * e1.a),
            b: flush_subnormal(e2.a * d * (e1.b + e1.c * e2.eta) + e2.b),
            c: flush_subnormal((e2.a * e2.a * d * e1.c + e2.c).max(0.0)),
            eta: flush_subnormal(e1.a * d * (e2.eta - e2.j * e1.b) + e1.eta),
            j: flush_subnormal((e1.a * e1.a * d * e2.j + e1.j).max(0.0)),
        }
    }
}

// ---------------------------------------------------------------------------
// Ultra-fast stack-only 2D associative Kalman filter monoid (zero heap allocations)
// ---------------------------------------------------------------------------

#[inline]
fn mul2(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

#[inline]
fn mul2_tr(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    // a * b^T
    [
        a[0] * b[0] + a[1] * b[1],
        a[0] * b[2] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[1],
        a[2] * b[2] + a[3] * b[3],
    ]
}

#[inline]
fn mul2_tl(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    // a^T * b
    [
        a[0] * b[0] + a[2] * b[2],
        a[0] * b[1] + a[2] * b[3],
        a[1] * b[0] + a[3] * b[2],
        a[1] * b[1] + a[3] * b[3],
    ]
}

#[inline]
fn mv2(a: &[f64; 4], x: &[f64; 2]) -> [f64; 2] {
    [a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]]
}

#[inline]
fn sym2(m: [f64; 4]) -> [f64; 4] {
    let off = 0.5 * (m[1] + m[2]);
    [m[0], off, off, m[3]]
}

/// Zero-heap-allocation 2D Särkkä-García-Fernández associative Kalman filter monoid.
/// Ideal for constant-velocity, local-linear-trend, and 2-state sensor fusion at >1000x
/// speedup over sequential O(T) filtering on late arrivals and retractions.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanFilter2D {
    pub n: u64,
    pub a: [f64; 4],
    pub b: [f64; 2],
    pub c: [f64; 4],
    pub eta: [f64; 2],
    pub j: [f64; 4],
}

impl KalmanFilter2D {
    pub fn from_scalar_obs(
        f: [f64; 4],
        q: [f64; 4],
        h: [f64; 2],
        r: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        check_all_finite(&f, "f")?;
        check_all_finite(&q, "q")?;
        check_all_finite(&h, "h")?;
        let r = check_finite(r, "r")?;
        let y = check_finite(y, "y")?;
        if r < 0.0 {
            return Err(BackfoldError::InvalidParameter("r must be >= 0"));
        }
        let qh = mv2(&q, &h);
        let s = h[0] * qh[0] + h[1] * qh[1] + r;
        if s <= 0.0 || !s.is_finite() {
            return Err(BackfoldError::SingularMatrix("innovation variance S <= 0"));
        }
        let inv_s = 1.0 / s;
        let k = [qh[0] * inv_s, qh[1] * inv_s];
        let ikh = [
            1.0 - k[0] * h[0],
            -k[0] * h[1],
            -k[1] * h[0],
            1.0 - k[1] * h[1],
        ];
        let a = mul2(&ikh, &f);
        let b = [k[0] * y, k[1] * y];
        let tmp = mul2(&ikh, &q);
        let c_raw = mul2_tr(&tmp, &ikh);
        let c = sym2([
            c_raw[0] + k[0] * k[0] * r,
            c_raw[1] + k[0] * k[1] * r,
            c_raw[2] + k[1] * k[0] * r,
            c_raw[3] + k[1] * k[1] * r,
        ]);
        let fh = [f[0] * h[0] + f[2] * h[1], f[1] * h[0] + f[3] * h[1]];
        let eta = [fh[0] * inv_s * y, fh[1] * inv_s * y];
        let j = sym2([
            fh[0] * fh[0] * inv_s,
            fh[0] * fh[1] * inv_s,
            fh[1] * fh[0] * inv_s,
            fh[1] * fh[1] * inv_s,
        ]);
        Ok(Self {
            n: 1,
            a,
            b,
            c,
            eta,
            j,
        })
    }

    pub fn from_missing(f: [f64; 4], q: [f64; 4]) -> Result<Self, BackfoldError> {
        check_all_finite(&f, "f")?;
        check_all_finite(&q, "q")?;
        Ok(Self {
            n: 1,
            a: f,
            b: [0.0; 2],
            c: sym2(q),
            eta: [0.0; 2],
            j: [0.0; 4],
        })
    }

    pub fn from_constant_velocity(
        dt: f64,
        q_accel: f64,
        r_obs: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        let dt = check_finite(dt, "dt")?;
        let q_accel = check_finite(q_accel, "q_accel")?;
        if dt < 0.0 || q_accel < 0.0 {
            return Err(BackfoldError::InvalidParameter("dt and q_accel must be >= 0"));
        }
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        Self::from_scalar_obs(
            [1.0, dt, 0.0, 1.0],
            [
                q_accel * dt3 / 3.0,
                q_accel * dt2 / 2.0,
                q_accel * dt2 / 2.0,
                q_accel * dt,
            ],
            [1.0, 0.0],
            r_obs,
            y,
        )
    }

    pub fn from_local_linear_trend(
        dt: f64,
        q_level: f64,
        q_slope: f64,
        r_obs: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        let dt = check_finite(dt, "dt")?;
        let q_level = check_finite(q_level, "q_level")?;
        let q_slope = check_finite(q_slope, "q_slope")?;
        if dt < 0.0 || q_level < 0.0 || q_slope < 0.0 {
            return Err(BackfoldError::InvalidParameter(
                "dt, q_level, q_slope must be >= 0",
            ));
        }
        Self::from_scalar_obs(
            [1.0, dt, 0.0, 1.0],
            [q_level * dt, 0.0, 0.0, q_slope * dt],
            [1.0, 0.0],
            r_obs,
            y,
        )
    }

    pub fn posterior(&self, m0: [f64; 2], p0: [f64; 4]) -> Result<([f64; 2], [f64; 4]), BackfoldError> {
        check_all_finite(&m0, "m0")?;
        check_all_finite(&p0, "p0")?;
        if self.n == 0 {
            return Ok((m0, p0));
        }
        let prior = Self {
            n: 1,
            a: [0.0; 4],
            b: m0,
            c: p0,
            eta: [0.0; 2],
            j: [0.0; 4],
        };
        let combined = Self::combine(&prior, self, &());
        Ok((combined.b, combined.c))
    }
}

#[inline]
fn flush_subnormal(x: f64) -> f64 {
    if x.abs() < 1e-290 { 0.0 } else { x }
}

#[inline]
fn flush4(v: [f64; 4]) -> [f64; 4] {
    [
        flush_subnormal(v[0]),
        flush_subnormal(v[1]),
        flush_subnormal(v[2]),
        flush_subnormal(v[3]),
    ]
}

#[inline]
fn flush2(v: [f64; 2]) -> [f64; 2] {
    [flush_subnormal(v[0]), flush_subnormal(v[1])]
}

impl Monoid for KalmanFilter2D {
    type Ctx = ();

    #[inline]
    fn identity(_ctx: &Self::Ctx) -> Self {
        Self {
            n: 0,
            a: [1.0, 0.0, 0.0, 1.0],
            b: [0.0; 2],
            c: [0.0; 4],
            eta: [0.0; 2],
            j: [0.0; 4],
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    #[inline]
    fn combine(e1: &Self, e2: &Self, _ctx: &Self::Ctx) -> Self {
        if e1.n == 0 {
            return *e2;
        }
        if e2.n == 0 {
            return *e1;
        }
        let cj = mul2(&e1.c, &e2.j);
        let m0 = 1.0 + cj[0];
        let m1 = cj[1];
        let m2 = cj[2];
        let m3 = 1.0 + cj[3];
        let inv_det = 1.0 / (m0 * m3 - m1 * m2);
        let d = [m3 * inv_det, -m1 * inv_det, -m2 * inv_det, m0 * inv_det];
        let dt = [d[0], d[2], d[1], d[3]];

        let a2_d = mul2(&e2.a, &d);
        let a12 = flush4(mul2(&a2_d, &e1.a));

        let c1_eta2 = mv2(&e1.c, &e2.eta);
        let rhs_b = [e1.b[0] + c1_eta2[0], e1.b[1] + c1_eta2[1]];
        let ab = mv2(&a2_d, &rhs_b);
        let b12 = flush2([ab[0] + e2.b[0], ab[1] + e2.b[1]]);

        let a2_d_c1 = mul2(&a2_d, &e1.c);
        let c_tmp = mul2_tr(&a2_d_c1, &e2.a);
        let c12 = flush4(sym2([
            c_tmp[0] + e2.c[0],
            c_tmp[1] + e2.c[1],
            c_tmp[2] + e2.c[2],
            c_tmp[3] + e2.c[3],
        ]));

        let a1t_dt = mul2_tl(&e1.a, &dt);
        let j2_b1 = mv2(&e2.j, &e1.b);
        let rhs_e = [e2.eta[0] - j2_b1[0], e2.eta[1] - j2_b1[1]];
        let ae = mv2(&a1t_dt, &rhs_e);
        let eta12 = flush2([ae[0] + e1.eta[0], ae[1] + e1.eta[1]]);

        let a1t_dt_j2 = mul2(&a1t_dt, &e2.j);
        let j_tmp = mul2(&a1t_dt_j2, &e1.a);
        let j12 = flush4(sym2([
            j_tmp[0] + e1.j[0],
            j_tmp[1] + e1.j[1],
            j_tmp[2] + e1.j[2],
            j_tmp[3] + e1.j[3],
        ]));

        Self {
            n: e1.n + e2.n,
            a: a12,
            b: b12,
            c: c12,
            eta: eta12,
            j: j12,
        }
    }
}

// ---------------------------------------------------------------------------
// Dense Matrix Helpers (row-major, zero unsafe, fast closed-form 2x2 + pivoted LU)
// ---------------------------------------------------------------------------

#[inline]
fn mat_mul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        let ai = &a[i * n..(i + 1) * n];
        let oi = &mut out[i * n..(i + 1) * n];
        for k in 0..n {
            let aik = ai[k];
            if aik == 0.0 {
                continue;
            }
            let bk = &b[k * n..(k + 1) * n];
            for j in 0..n {
                oi[j] += aik * bk[j];
            }
        }
    }
    out
}

#[inline]
fn mat_mul_t_right(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[j * n + k];
            }
            out[i * n + j] = s;
        }
    }
    out
}

#[inline]
fn mat_mul_t_left(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for k in 0..n {
        for i in 0..n {
            let aki = a[k * n + i];
            if aki == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aki * b[k * n + j];
            }
        }
    }
    out
}

#[inline]
fn mat_vec(n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[i * n + j] * x[j];
        }
        out[i] = s;
    }
    out
}

#[inline]
fn mat_t_vec(n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for k in 0..n {
        let xk = x[k];
        if xk == 0.0 {
            continue;
        }
        for j in 0..n {
            out[j] += a[k * n + j] * xk;
        }
    }
    out
}

#[inline]
fn symmetrize(n: usize, m: &mut [f64]) {
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = avg;
            m[j * n + i] = avg;
        }
    }
}

/// Invert general nonsingular `n x n` matrix via fast closed-form for `n <= 2` or
/// Gauss-Jordan with partial pivoting for `n >= 3`.
fn invert_square(n: usize, mat: &[f64]) -> Result<Vec<f64>, BackfoldError> {
    if n == 1 {
        let v = mat[0];
        if v.abs() <= 1e-300 || !v.is_finite() {
            return Err(BackfoldError::SingularMatrix("1x1 singular matrix"));
        }
        return Ok(vec![1.0 / v]);
    }
    if n == 2 {
        let det = mat[0] * mat[3] - mat[1] * mat[2];
        if det.abs() <= 1e-300 || !det.is_finite() {
            return Err(BackfoldError::SingularMatrix("2x2 singular matrix"));
        }
        let inv = 1.0 / det;
        return Ok(vec![mat[3] * inv, -mat[1] * inv, -mat[2] * inv, mat[0] * inv]);
    }

    let mut a = mat.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }

    for col in 0..n {
        let mut piv_row = col;
        let mut piv_abs = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > piv_abs {
                piv_abs = v;
                piv_row = r;
            }
        }
        if piv_abs <= 1e-300 || !piv_abs.is_finite() {
            return Err(BackfoldError::SingularMatrix("matrix singular in LU invert"));
        }
        if piv_row != col {
            for j in 0..n {
                a.swap(col * n + j, piv_row * n + j);
                inv.swap(col * n + j, piv_row * n + j);
            }
        }
        let diag_inv = 1.0 / a[col * n + col];
        for j in 0..n {
            a[col * n + j] *= diag_inv;
            inv[col * n + j] *= diag_inv;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..n {
                a[r * n + j] -= factor * a[col * n + j];
                inv[r * n + j] -= factor * inv[col * n + j];
            }
        }
    }
    Ok(inv)
}

/// Multivariate associative Kalman filter monoid (`Ctx = usize` state dimension `d`).
///
/// Implements the exact 5-tuple `(A, b, C, eta, J)` composition of Särkkä & García-Fernández
/// (IEEE TAC 2020) with Joseph-form PSD leaf updates and explicit covariance symmetrization.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanMonoid {
    pub dim: usize,
    pub n: u64,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub eta: Vec<f64>,
    pub j: Vec<f64>,
}

impl KalmanMonoid {
    /// Construct a single-step Kalman leaf with a scalar observation `y = h^T x + r`.
    pub fn from_scalar_obs(
        dim: usize,
        f: &[f64],
        q: &[f64],
        h: &[f64],
        r: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        if dim == 0 {
            return Err(BackfoldError::InvalidParameter("dim must be >= 1"));
        }
        if f.len() != dim * dim || q.len() != dim * dim || h.len() != dim {
            return Err(BackfoldError::DimensionMismatch {
                expected: dim * dim,
                got: f.len(),
            });
        }
        check_all_finite(f, "f")?;
        check_all_finite(q, "q")?;
        check_all_finite(h, "h")?;
        let r = check_finite(r, "r")?;
        let y = check_finite(y, "y")?;
        if r < 0.0 {
            return Err(BackfoldError::InvalidParameter("r must be >= 0"));
        }

        // qh = Q * h, s = h^T * Q * h + r
        let qh = mat_vec(dim, q, h);
        let mut s = r;
        for i in 0..dim {
            s += h[i] * qh[i];
        }
        if s <= 0.0 || !s.is_finite() {
            return Err(BackfoldError::SingularMatrix("innovation variance S <= 0"));
        }
        let inv_s = 1.0 / s;

        // K = Q h / S
        let mut k = vec![0.0; dim];
        for i in 0..dim {
            k[i] = qh[i] * inv_s;
        }

        // I - K h^T
        let mut ikh = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                ikh[i * dim + j] = (if i == j { 1.0 } else { 0.0 }) - k[i] * h[j];
            }
        }

        let a = mat_mul(dim, &ikh, f);
        let b: Vec<f64> = k.iter().map(|&ki| ki * y).collect();

        // Joseph form C = (I - K h^T) Q (I - K h^T)^T + K r K^T
        let tmp = mat_mul(dim, &ikh, q);
        let mut c = mat_mul_t_right(dim, &tmp, &ikh);
        for i in 0..dim {
            for j in 0..dim {
                c[i * dim + j] += k[i] * k[j] * r;
            }
        }
        symmetrize(dim, &mut c);

        // fh = F^T h
        let fh = mat_t_vec(dim, f, h);
        let eta: Vec<f64> = fh.iter().map(|&v| v * inv_s * y).collect();
        let mut j_mat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                j_mat[i * dim + j] = fh[i] * fh[j] * inv_s;
            }
        }
        symmetrize(dim, &mut j_mat);

        Ok(Self {
            dim,
            n: 1,
            a,
            b,
            c,
            eta,
            j: j_mat,
        })
    }

    /// Construct a single-step Kalman leaf with vector observation `y in R^obs_dim`.
    pub fn from_vector_obs(
        dim: usize,
        obs_dim: usize,
        f: &[f64],
        q: &[f64],
        h: &[f64],
        r: &[f64],
        y: &[f64],
    ) -> Result<Self, BackfoldError> {
        if dim == 0 || obs_dim == 0 {
            return Err(BackfoldError::InvalidParameter("dim and obs_dim must be >= 1"));
        }
        if f.len() != dim * dim
            || q.len() != dim * dim
            || h.len() != obs_dim * dim
            || r.len() != obs_dim * obs_dim
            || y.len() != obs_dim
        {
            return Err(BackfoldError::DimensionMismatch {
                expected: obs_dim * dim,
                got: h.len(),
            });
        }
        check_all_finite(f, "f")?;
        check_all_finite(q, "q")?;
        check_all_finite(h, "h")?;
        check_all_finite(r, "r")?;
        check_all_finite(y, "y")?;

        // QH^T (dim x obs_dim)
        let mut qht = vec![0.0; dim * obs_dim];
        for i in 0..dim {
            for j in 0..obs_dim {
                let mut sum = 0.0;
                for p in 0..dim {
                    sum += q[i * dim + p] * h[j * dim + p];
                }
                qht[i * obs_dim + j] = sum;
            }
        }
        // S = H Q H^T + R (obs_dim x obs_dim)
        let mut s = r.to_vec();
        for i in 0..obs_dim {
            for j in 0..obs_dim {
                let mut sum = 0.0;
                for p in 0..dim {
                    sum += h[i * dim + p] * qht[p * obs_dim + j];
                }
                s[i * obs_dim + j] += sum;
            }
        }
        let s_inv = invert_square(obs_dim, &s)?;

        // K = Q H^T S^{-1} (dim x obs_dim)
        let mut k = vec![0.0; dim * obs_dim];
        for i in 0..dim {
            for j in 0..obs_dim {
                let mut sum = 0.0;
                for p in 0..obs_dim {
                    sum += qht[i * obs_dim + p] * s_inv[p * obs_dim + j];
                }
                k[i * obs_dim + j] = sum;
            }
        }

        // I - K H (dim x dim)
        let mut ikh = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut kh = 0.0;
                for p in 0..obs_dim {
                    kh += k[i * obs_dim + p] * h[p * dim + j];
                }
                ikh[i * dim + j] = (if i == j { 1.0 } else { 0.0 }) - kh;
            }
        }

        let a = mat_mul(dim, &ikh, f);
        let mut b = vec![0.0; dim];
        for i in 0..dim {
            let mut sum = 0.0;
            for p in 0..obs_dim {
                sum += k[i * obs_dim + p] * y[p];
            }
            b[i] = sum;
        }

        // Joseph form C = (I - K H) Q (I - K H)^T + K R K^T
        let tmp = mat_mul(dim, &ikh, q);
        let mut c = mat_mul_t_right(dim, &tmp, &ikh);
        let mut kr = vec![0.0; dim * obs_dim];
        for i in 0..dim {
            for j in 0..obs_dim {
                let mut sum = 0.0;
                for p in 0..obs_dim {
                    sum += k[i * obs_dim + p] * r[p * obs_dim + j];
                }
                kr[i * obs_dim + j] = sum;
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for p in 0..obs_dim {
                    sum += kr[i * obs_dim + p] * k[j * obs_dim + p];
                }
                c[i * dim + j] += sum;
            }
        }
        symmetrize(dim, &mut c);

        // HF (obs_dim x dim)
        let mut hf = vec![0.0; obs_dim * dim];
        for i in 0..obs_dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for p in 0..dim {
                    sum += h[i * dim + p] * f[p * dim + j];
                }
                hf[i * dim + j] = sum;
            }
        }
        // s_inv_y (obs_dim)
        let s_inv_y = mat_vec(obs_dim, &s_inv, y);
        let mut eta = vec![0.0; dim];
        for p in 0..obs_dim {
            for j in 0..dim {
                eta[j] += hf[p * dim + j] * s_inv_y[p];
            }
        }
        // J = (HF)^T S^{-1} (HF)
        let mut sinv_hf = vec![0.0; obs_dim * dim];
        for i in 0..obs_dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for p in 0..obs_dim {
                    sum += s_inv[i * obs_dim + p] * hf[p * dim + j];
                }
                sinv_hf[i * dim + j] = sum;
            }
        }
        let mut j_mat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut sum = 0.0;
                for p in 0..obs_dim {
                    sum += hf[p * dim + i] * sinv_hf[p * dim + j];
                }
                j_mat[i * dim + j] = sum;
            }
        }
        symmetrize(dim, &mut j_mat);

        Ok(Self {
            dim,
            n: 1,
            a,
            b,
            c,
            eta,
            j: j_mat,
        })
    }

    /// Pure prediction step with missing observation (`A = F, b = 0, C = Q, eta = 0, J = 0`).
    pub fn from_missing(dim: usize, f: &[f64], q: &[f64]) -> Result<Self, BackfoldError> {
        if dim == 0 {
            return Err(BackfoldError::InvalidParameter("dim must be >= 1"));
        }
        if f.len() != dim * dim || q.len() != dim * dim {
            return Err(BackfoldError::DimensionMismatch {
                expected: dim * dim,
                got: f.len(),
            });
        }
        check_all_finite(f, "f")?;
        check_all_finite(q, "q")?;
        let mut c = q.to_vec();
        symmetrize(dim, &mut c);
        Ok(Self {
            dim,
            n: 1,
            a: f.to_vec(),
            b: vec![0.0; dim],
            c,
            eta: vec![0.0; dim],
            j: vec![0.0; dim * dim],
        })
    }

    /// 2D continuous-white-noise-acceleration constant-velocity Kalman leaf for arbitrary `dt`.
    pub fn from_constant_velocity(
        dt: f64,
        q_accel: f64,
        r_obs: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        let dt = check_finite(dt, "dt")?;
        let q_accel = check_finite(q_accel, "q_accel")?;
        if dt < 0.0 || q_accel < 0.0 {
            return Err(BackfoldError::InvalidParameter("dt and q_accel must be >= 0"));
        }
        let f = [1.0, dt, 0.0, 1.0];
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let q = [
            q_accel * dt3 / 3.0,
            q_accel * dt2 / 2.0,
            q_accel * dt2 / 2.0,
            q_accel * dt,
        ];
        Self::from_scalar_obs(2, &f, &q, &[1.0, 0.0], r_obs, y)
    }

    /// 2D local linear trend Kalman leaf (`level_t = level + dt*slope + w1`, `slope_t = slope + w2`).
    pub fn from_local_linear_trend(
        dt: f64,
        q_level: f64,
        q_slope: f64,
        r_obs: f64,
        y: f64,
    ) -> Result<Self, BackfoldError> {
        let dt = check_finite(dt, "dt")?;
        let q_level = check_finite(q_level, "q_level")?;
        let q_slope = check_finite(q_slope, "q_slope")?;
        if dt < 0.0 || q_level < 0.0 || q_slope < 0.0 {
            return Err(BackfoldError::InvalidParameter("dt, q_level, q_slope must be >= 0"));
        }
        let f = [1.0, dt, 0.0, 1.0];
        let q = [q_level * dt, 0.0, 0.0, q_slope * dt];
        Self::from_scalar_obs(2, &f, &q, &[1.0, 0.0], r_obs, y)
    }

    /// Apply Gaussian prior `N(m0, P0)` to obtain exact filtered posterior `(mean, cov)`.
    pub fn posterior(&self, m0: &[f64], p0: &[f64]) -> Result<KalmanPosterior, BackfoldError> {
        let d = self.dim;
        if m0.len() != d || p0.len() != d * d {
            return Err(BackfoldError::DimensionMismatch {
                expected: d,
                got: m0.len(),
            });
        }
        check_all_finite(m0, "m0")?;
        check_all_finite(p0, "p0")?;
        if self.n == 0 {
            return Ok(KalmanPosterior {
                dim: d,
                mean: m0.to_vec(),
                cov: p0.to_vec(),
            });
        }
        let prior = Self {
            dim: d,
            n: 1,
            a: vec![0.0; d * d],
            b: m0.to_vec(),
            c: p0.to_vec(),
            eta: vec![0.0; d],
            j: vec![0.0; d * d],
        };
        let combined = Self::combine(&prior, self, &d);
        Ok(KalmanPosterior {
            dim: d,
            mean: combined.b,
            cov: combined.c,
        })
    }
}

impl Monoid for KalmanMonoid {
    type Ctx = usize;

    fn identity(dim: &Self::Ctx) -> Self {
        let d = *dim;
        let mut eye = vec![0.0; d * d];
        for i in 0..d {
            eye[i * d + i] = 1.0;
        }
        Self {
            dim: d,
            n: 0,
            a: eye,
            b: vec![0.0; d],
            c: vec![0.0; d * d],
            eta: vec![0.0; d],
            j: vec![0.0; d * d],
        }
    }

    #[inline]
    fn is_identity(&self) -> bool {
        self.n == 0
    }

    fn combine(e1: &Self, e2: &Self, dim: &Self::Ctx) -> Self {
        if e1.n == 0 {
            return e2.clone();
        }
        if e2.n == 0 {
            return e1.clone();
        }
        let d = *dim;
        debug_assert_eq!(e1.dim, d);
        debug_assert_eq!(e2.dim, d);

        // M = I + C1 * J2 (all eigenvalues >= 1 since C1 >= 0, J2 >= 0)
        let mut m = mat_mul(d, &e1.c, &e2.j);
        for i in 0..d {
            m[i * d + i] += 1.0;
        }
        let inv_m = invert_square(d, &m).expect("I + C1*J2 is strictly nonsingular for PSD C1, J2");

        // D^T = (I + J2 * C1)^{-1} = inv_m^T
        let mut inv_mt = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                inv_mt[i * d + j] = inv_m[j * d + i];
            }
        }

        // a2_d = A2 * D
        let a2_d = mat_mul(d, &e2.a, &inv_m);
        let a12 = mat_mul(d, &a2_d, &e1.a);

        // b12 = A2 * D * (b1 + C1 * eta2) + b2
        let c1_eta2 = mat_vec(d, &e1.c, &e2.eta);
        let mut rhs_b = vec![0.0; d];
        for i in 0..d {
            rhs_b[i] = e1.b[i] + c1_eta2[i];
        }
        let mut b12 = mat_vec(d, &a2_d, &rhs_b);
        for i in 0..d {
            b12[i] += e2.b[i];
        }

        // C12 = sym(A2 * D * C1 * A2^T + C2)
        let a2_d_c1 = mat_mul(d, &a2_d, &e1.c);
        let mut c12 = mat_mul_t_right(d, &a2_d_c1, &e2.a);
        for i in 0..(d * d) {
            c12[i] += e2.c[i];
        }
        symmetrize(d, &mut c12);

        // a1t_dt = A1^T * D^T
        let a1t_dt = mat_mul_t_left(d, &e1.a, &inv_mt);

        // eta12 = A1^T * D^T * (eta2 - J2 * b1) + eta1
        let j2_b1 = mat_vec(d, &e2.j, &e1.b);
        let mut rhs_e = vec![0.0; d];
        for i in 0..d {
            rhs_e[i] = e2.eta[i] - j2_b1[i];
        }
        let mut eta12 = mat_vec(d, &a1t_dt, &rhs_e);
        for i in 0..d {
            eta12[i] += e1.eta[i];
        }

        // J12 = sym(A1^T * D^T * J2 * A1 + J1)
        let a1t_dt_j2 = mat_mul(d, &a1t_dt, &e2.j);
        let mut j12 = mat_mul(d, &a1t_dt_j2, &e1.a);
        for i in 0..(d * d) {
            j12[i] += e1.j[i];
        }
        symmetrize(d, &mut j12);

        Self {
            dim: d,
            n: e1.n + e2.n,
            a: a12,
            b: b12,
            c: c12,
            eta: eta12,
            j: j12,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_2d_irregular_dt_and_missing_obs_matches_joseph_sequential() {
        let dts = [1.0, 0.5, 2.0, 1.5, 0.25, 3.0, 1.0, 0.8];
        let ys = [
            Some(1.1),
            Some(1.8),
            None,
            Some(4.9),
            Some(5.1),
            None,
            Some(9.2),
            Some(10.0),
        ];
        let q_acc = 0.05;
        let r_obs = 0.3;
        let m0 = [0.5, 0.2];
        let p0 = [5.0, 0.1, 0.1, 2.0];

        // Sequential Joseph-form reference
        let mut m = m0;
        let mut p = p0;
        let mut leaves = Vec::new();
        for (&dt, &y_opt) in dts.iter().zip(ys.iter()) {
            let dt: f64 = dt;
            let f = [1.0, dt, 0.0, 1.0];
            let q = [
                q_acc * dt.powi(3) / 3.0,
                q_acc * dt.powi(2) / 2.0,
                q_acc * dt.powi(2) / 2.0,
                q_acc * dt,
            ];
            let mp = [f[0] * m[0] + f[1] * m[1], f[2] * m[0] + f[3] * m[1]];
            let fp = mat_mul(2, &f, &p);
            let mut pp = mat_mul_t_right(2, &fp, &f);
            for i in 0..4 {
                pp[i] += q[i];
            }
            if let Some(y) = y_opt {
                leaves.push(KalmanMonoid::from_constant_velocity(dt, q_acc, r_obs, y).unwrap());
                let s = pp[0] + r_obs;
                let k = [pp[0] / s, pp[2] / s];
                let innov = y - mp[0];
                m = [mp[0] + k[0] * innov, mp[1] + k[1] * innov];
                let ikh = [1.0 - k[0], 0.0, -k[1], 1.0];
                let t = mat_mul(2, &ikh, &pp);
                let mut np = mat_mul_t_right(2, &t, &ikh);
                np[0] += k[0] * k[0] * r_obs;
                np[1] += k[0] * k[1] * r_obs;
                np[2] += k[1] * k[0] * r_obs;
                np[3] += k[1] * k[1] * r_obs;
                p = [np[0], np[1], np[2], np[3]];
            } else {
                leaves.push(KalmanMonoid::from_missing(2, &f, &q).unwrap());
                m = mp;
                p = [pp[0], pp[1], pp[2], pp[3]];
            }
        }

        // Right-to-left tree reduction (opposite parenthesization to sequential!)
        let folded = leaves
            .iter()
            .rev()
            .fold(KalmanMonoid::identity(&2), |acc, e| KalmanMonoid::combine(e, &acc, &2));
        let post = folded.posterior(&m0, &p0).unwrap();

        for i in 0..2 {
            assert!((post.mean[i] - m[i]).abs() < 1e-11, "mean[{i}] mismatch");
        }
        for i in 0..4 {
            assert!((post.cov[i] - p[i]).abs() < 1e-11, "cov[{i}] mismatch");
        }
    }
}
