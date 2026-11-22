//! Exact Särkkä & García-Fernández (2020, IEEE TAC) associative linear Gaussian Kalman filter.
//!
//! Enables O(log T) out-of-order sensor insertions, retractions, retroactive corrections,
//! missing observations, irregular `dt` continuous-time transitions, and O(1) counterfactual forks
//! with bit-tight parity against Joseph-form sequential Kalman filtering.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::key::EventKey;
use crate::monoid::{check_all_finite, check_finite, BackfoldError, Monoid};
use crate::tree::FoldTree;

const LN_TAU: f64 = 1.837_877_066_409_345_3; // ln(2 * pi)

/// Exact posterior state `(mean, cov)` at the end of a filtered or smoothed interval.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanPosterior {
    pub dim: usize,
    pub mean: Vec<f64>,
    pub cov: Vec<f64>,
}

/// Ultra-fast stack-only scalar (1D) associative Kalman filter/smoother monoid (zero heap allocations).
///
/// Uses anchor-centered Gaussian potentials around `anchor` so `log_z` has zero `O(y^2 / S)`
/// catastrophic cancellation even on drifting or high-offset (`1e6+`) trajectories.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanFilter1D {
    pub n: u64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub eta: f64,
    pub j: f64,
    /// Input-state anchor `r` (`x_end = a*(x0 - r) + b`, `log ell(x0) = log_z + eta*(x0 - r) - 0.5*j*(x0 - r)^2`).
    pub anchor: f64,
    /// Exact anchor-centered Gaussian potential log-constant enabling O(log T) marginal log-likelihood.
    pub log_z: f64,
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
        let g = f * h;
        let (anchor, b, eta, log_z) = if g.abs() > 1e-150 {
            let anc = y / g;
            (anc, f * anc, 0.0, -0.5 * (LN_TAU + s.ln()))
        } else {
            (0.0, k * y, 0.0, -0.5 * (LN_TAU + s.ln() + y * y * inv_s))
        };
        Ok(Self {
            n: 1,
            a: ikh * f,
            b,
            c: (ikh * ikh * q + k * k * r).max(0.0),
            eta,
            j: (g * g * inv_s).max(0.0),
            anchor,
            log_z,
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
            anchor: 0.0,
            log_z: 0.0,
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

    /// Apply initial Gaussian prior `N(m0, p0)` to obtain exact filtered posterior `(mean, var)`.
    pub fn posterior(&self, m0: f64, p0: f64) -> Result<(f64, f64), BackfoldError> {
        let m0 = check_finite(m0, "m0")?;
        let p0 = check_finite(p0, "p0")?;
        if p0 < 0.0 {
            return Err(BackfoldError::InvalidParameter("p0 must be >= 0"));
        }
        if self.n == 0 {
            return Ok((m0, p0));
        }
        let delta = m0 - self.anchor;
        let d = 1.0 / (1.0 + p0 * self.j);
        let mean = self.a * d * (delta + p0 * self.eta) + self.b;
        let var = (self.a * self.a * d * p0 + self.c).max(0.0);
        Ok((mean, var))
    }

    /// Prior-fusion increment `log p(y_{1:T} | N(m0, p0)) - self.log_z` (cancels `self.log_z`
    /// identically when comparing two priors on the same suffix, e.g. changepoint Bayes factors).
    #[inline]
    pub fn log_prior_coupling(&self, m0: f64, p0: f64) -> Result<f64, BackfoldError> {
        let m0 = check_finite(m0, "m0")?;
        let p0 = check_finite(p0, "p0")?;
        if p0 < 0.0 {
            return Err(BackfoldError::InvalidParameter("p0 must be >= 0"));
        }
        if self.n == 0 {
            return Ok(0.0);
        }
        let delta = m0 - self.anchor;
        let det = 1.0 + p0 * self.j;
        let d = 1.0 / det;
        let quad = delta * d * (2.0 * self.eta - self.j * delta) + self.eta * d * p0 * self.eta;
        Ok(-0.5 * det.ln() + 0.5 * quad)
    }

    /// Exact Bayesian marginal log-likelihood `log p(y_{1:T})` under initial prior `N(m0, p0)`.
    pub fn log_marginal_likelihood(&self, m0: f64, p0: f64) -> Result<f64, BackfoldError> {
        let coupling = self.log_prior_coupling(m0, p0)?;
        if self.n == 0 {
            Ok(0.0)
        } else {
            Ok(self.log_z + coupling)
        }
    }

    /// Exact Rauch-Tung-Striebel (RTS) Bayesian smoothing fusion combining filtered posterior
    /// `p(x_t | y_{1:t}) = N(m_filt, p_filt)` with the future suffix monoid `future` (`y_{t+1:T}`).
    pub fn smooth_fusion(m_filt: f64, p_filt: f64, future: &Self) -> Result<(f64, f64), BackfoldError> {
        let m_filt = check_finite(m_filt, "m_filt")?;
        let p_filt = check_finite(p_filt, "p_filt")?;
        if p_filt < 0.0 {
            return Err(BackfoldError::InvalidParameter("p_filt must be >= 0"));
        }
        if future.n == 0 {
            return Ok((m_filt, p_filt));
        }
        let d = 1.0 / (1.0 + p_filt * future.j);
        let m_smooth = m_filt + d * p_filt * (future.eta + future.j * (future.anchor - m_filt));
        let p_smooth = (d * p_filt).max(0.0);
        Ok((m_smooth, p_smooth))
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
            anchor: 0.0,
            log_z: 0.0,
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
        let delta = e1.b - e2.anchor;
        let m = 1.0 + e1.c * e2.j;
        let d = 1.0 / m;
        let quad = delta * d * (2.0 * e2.eta - e2.j * delta) + e2.eta * d * e1.c * e2.eta;
        let log_z = e1.log_z + e2.log_z - 0.5 * m.ln() + 0.5 * quad;
        Self {
            n: e1.n + e2.n,
            a: flush_subnormal(e2.a * d * e1.a),
            b: flush_subnormal(e2.a * d * (delta + e1.c * e2.eta) + e2.b),
            c: flush_subnormal((e2.a * e2.a * d * e1.c + e2.c).max(0.0)),
            eta: flush_subnormal(e1.a * d * (e2.eta - e2.j * delta) + e1.eta),
            j: flush_subnormal((e1.a * e1.a * d * e2.j + e1.j).max(0.0)),
            anchor: e1.anchor,
            log_z,
        }
    }
}

// ---------------------------------------------------------------------------
// Ultra-fast stack-only 2D associative Kalman filter/smoother monoid (zero heap allocations)
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
fn dot2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

#[inline]
fn sym2(m: [f64; 4]) -> [f64; 4] {
    let off = 0.5 * (m[1] + m[2]);
    [m[0], off, off, m[3]]
}

/// Zero-heap-allocation 2D Särkkä-García-Fernández associative Kalman filter & RTS smoother monoid
/// with anchor-centered Gaussian potentials for cancellation-free exact marginal log-likelihood.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KalmanFilter2D {
    pub n: u64,
    pub a: [f64; 4],
    pub b: [f64; 2],
    pub c: [f64; 4],
    pub eta: [f64; 2],
    pub j: [f64; 4],
    /// Input-state anchor `r` eliminating `O(y^2 / S)` cancellation in `log_z` and `smooth_fusion`.
    pub anchor: [f64; 2],
    /// Exact anchor-centered Gaussian potential log-constant enabling O(log T) marginal log-likelihood.
    pub log_z: f64,
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
        let tmp = mul2(&ikh, &q);
        let c_raw = mul2_tr(&tmp, &ikh);
        let c = sym2([
            c_raw[0] + k[0] * k[0] * r,
            c_raw[1] + k[0] * k[1] * r,
            c_raw[2] + k[1] * k[0] * r,
            c_raw[3] + k[1] * k[1] * r,
        ]);
        let fh = [f[0] * h[0] + f[2] * h[1], f[1] * h[0] + f[3] * h[1]];
        let j = sym2([
            fh[0] * fh[0] * inv_s,
            fh[0] * fh[1] * inv_s,
            fh[1] * fh[0] * inv_s,
            fh[1] * fh[1] * inv_s,
        ]);
        // Pick coordinate-aligned anchor satisfying fh^T anchor == y so log_z = -0.5*(ln 2pi + ln S),
        // eta = 0, and b = F * anchor (eliminating O(y^2/S) cancellation across 100k+ ticks!).
        let fh_max = fh[0].abs().max(fh[1].abs());
        let (anchor, b, eta, log_z) = if fh_max > 1e-150 {
            let anc = if fh[0].abs() > 1e-6 * fh_max {
                [y / fh[0], 0.0]
            } else {
                [0.0, y / fh[1]]
            };
            (anc, mv2(&f, &anc), [0.0, 0.0], -0.5 * (LN_TAU + s.ln()))
        } else {
            (
                [0.0, 0.0],
                [k[0] * y, k[1] * y],
                [0.0, 0.0],
                -0.5 * (LN_TAU + s.ln() + y * y * inv_s),
            )
        };
        Ok(Self {
            n: 1,
            a,
            b,
            c,
            eta,
            j,
            anchor,
            log_z,
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
            anchor: [0.0; 2],
            log_z: 0.0,
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
        let delta = [m0[0] - self.anchor[0], m0[1] - self.anchor[1]];
        let cj = mul2(&p0, &self.j);
        let det_m0 = 1.0 + cj[0];
        let det_m1 = cj[1];
        let det_m2 = cj[2];
        let det_m3 = 1.0 + cj[3];
        let inv_det = 1.0 / (det_m0 * det_m3 - det_m1 * det_m2);
        let d = [
            det_m3 * inv_det,
            -det_m1 * inv_det,
            -det_m2 * inv_det,
            det_m0 * inv_det,
        ];
        let a_d = mul2(&self.a, &d);
        let p_eta = mv2(&p0, &self.eta);
        let ab = mv2(&a_d, &[delta[0] + p_eta[0], delta[1] + p_eta[1]]);
        let mean = [ab[0] + self.b[0], ab[1] + self.b[1]];
        let a_d_p = mul2(&a_d, &p0);
        let c_tmp = mul2_tr(&a_d_p, &self.a);
        let cov = sym2([
            c_tmp[0] + self.c[0],
            c_tmp[1] + self.c[1],
            c_tmp[2] + self.c[2],
            c_tmp[3] + self.c[3],
        ]);
        Ok((mean, cov))
    }

    /// Prior-fusion increment `log p(y_{1:T} | N(m0, P0)) - self.log_z` (cancels `self.log_z`
    /// identically when comparing two priors on the same suffix, e.g. changepoint Bayes factors).
    #[inline]
    pub fn log_prior_coupling(&self, m0: [f64; 2], p0: [f64; 4]) -> Result<f64, BackfoldError> {
        check_all_finite(&m0, "m0")?;
        check_all_finite(&p0, "p0")?;
        if self.n == 0 {
            return Ok(0.0);
        }
        let delta = [m0[0] - self.anchor[0], m0[1] - self.anchor[1]];
        let cj = mul2(&p0, &self.j);
        let det_m0 = 1.0 + cj[0];
        let det_m1 = cj[1];
        let det_m2 = cj[2];
        let det_m3 = 1.0 + cj[3];
        let det = det_m0 * det_m3 - det_m1 * det_m2;
        let inv_det = 1.0 / det;
        let d = [
            det_m3 * inv_det,
            -det_m1 * inv_det,
            -det_m2 * inv_det,
            det_m0 * inv_det,
        ];
        let dt = [d[0], d[2], d[1], d[3]];
        let p_eta = mv2(&p0, &self.eta);
        let j_delta = mv2(&self.j, &delta);
        let rhs_e = [self.eta[0] - j_delta[0], self.eta[1] - j_delta[1]];
        let dt_v = mv2(&dt, &[self.eta[0] + rhs_e[0], self.eta[1] + rhs_e[1]]);
        let d_p_eta = mv2(&d, &p_eta);
        let quad = dot2(&delta, &dt_v) + dot2(&self.eta, &d_p_eta);
        Ok(-0.5 * det.abs().ln() + 0.5 * quad)
    }

    /// Exact Bayesian marginal log-likelihood `log p(y_{1:T})` under initial prior `N(m0, P0)`.
    pub fn log_marginal_likelihood(&self, m0: [f64; 2], p0: [f64; 4]) -> Result<f64, BackfoldError> {
        let coupling = self.log_prior_coupling(m0, p0)?;
        if self.n == 0 {
            Ok(0.0)
        } else {
            Ok(self.log_z + coupling)
        }
    }

    /// Exact Rauch-Tung-Striebel (RTS) Bayesian smoothing fusion combining filtered posterior
    /// `p(x_t | y_{1:t}) = N(m_filt, p_filt)` with the future suffix monoid `future` (`y_{t+1:T}`).
    pub fn smooth_fusion(
        m_filt: [f64; 2],
        p_filt: [f64; 4],
        future: &Self,
    ) -> Result<([f64; 2], [f64; 4]), BackfoldError> {
        check_all_finite(&m_filt, "m_filt")?;
        check_all_finite(&p_filt, "p_filt")?;
        if future.n == 0 {
            return Ok((m_filt, p_filt));
        }
        let pj = mul2(&p_filt, &future.j);
        let m0 = 1.0 + pj[0];
        let m1 = pj[1];
        let m2 = pj[2];
        let m3 = 1.0 + pj[3];
        let inv_det = 1.0 / (m0 * m3 - m1 * m2);
        let d = [m3 * inv_det, -m1 * inv_det, -m2 * inv_det, m0 * inv_det];

        // Centered correction: m_smooth = m_filt + D * P_filt * (eta + J * (anchor - m_filt))
        let diff = [future.anchor[0] - m_filt[0], future.anchor[1] - m_filt[1]];
        let j_diff = mv2(&future.j, &diff);
        let p_rhs = mv2(&p_filt, &[future.eta[0] + j_diff[0], future.eta[1] + j_diff[1]]);
        let corr = mv2(&d, &p_rhs);
        let m_smooth = [m_filt[0] + corr[0], m_filt[1] + corr[1]];
        let p_smooth = sym2(mul2(&d, &p_filt));
        Ok((m_smooth, p_smooth))
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
            anchor: [0.0; 2],
            log_z: 0.0,
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
        let delta = [e1.b[0] - e2.anchor[0], e1.b[1] - e2.anchor[1]];
        let cj = mul2(&e1.c, &e2.j);
        let m0 = 1.0 + cj[0];
        let m1 = cj[1];
        let m2 = cj[2];
        let m3 = 1.0 + cj[3];
        let det = m0 * m3 - m1 * m2;
        let inv_det = 1.0 / det;
        let d = [m3 * inv_det, -m1 * inv_det, -m2 * inv_det, m0 * inv_det];
        let dt = [d[0], d[2], d[1], d[3]];

        let a2_d = mul2(&e2.a, &d);
        let a12 = flush4(mul2(&a2_d, &e1.a));

        let c1_eta2 = mv2(&e1.c, &e2.eta);
        let rhs_b = [delta[0] + c1_eta2[0], delta[1] + c1_eta2[1]];
        let ab = mv2(&a2_d, &rhs_b);
        let b12 = [ab[0] + e2.b[0], ab[1] + e2.b[1]];

        let a2_d_c1 = mul2(&a2_d, &e1.c);
        let c_tmp = mul2_tr(&a2_d_c1, &e2.a);
        let c12 = sym2([
            c_tmp[0] + e2.c[0],
            c_tmp[1] + e2.c[1],
            c_tmp[2] + e2.c[2],
            c_tmp[3] + e2.c[3],
        ]);

        let a1t_dt = mul2_tl(&e1.a, &dt);
        let j2_delta = mv2(&e2.j, &delta);
        let rhs_e = [e2.eta[0] - j2_delta[0], e2.eta[1] - j2_delta[1]];
        let ae = mv2(&a1t_dt, &rhs_e);
        let eta12 = flush2([ae[0] + e1.eta[0], ae[1] + e1.eta[1]]);

        let a1t_dt_j2 = mul2(&a1t_dt, &e2.j);
        let j_tmp = mul2(&a1t_dt_j2, &e1.a);
        let j12 = sym2([
            j_tmp[0] + e1.j[0],
            j_tmp[1] + e1.j[1],
            j_tmp[2] + e1.j[2],
            j_tmp[3] + e1.j[3],
        ]);

        // Exact anchor-centered associative marginal log-likelihood update (delta = O(1)!)
        let dt_v = mv2(&dt, &[e2.eta[0] + rhs_e[0], e2.eta[1] + rhs_e[1]]);
        let d_c1_eta2 = mv2(&d, &c1_eta2);
        let quad = dot2(&delta, &dt_v) + dot2(&e2.eta, &d_c1_eta2);
        let log_z = e1.log_z + e2.log_z - 0.5 * det.abs().ln() + 0.5 * quad;

        Self {
            n: e1.n + e2.n,
            a: a12,
            b: b12,
            c: c12,
            eta: eta12,
            j: j12,
            anchor: e1.anchor,
            log_z,
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

/// Invert general nonsingular `n x n` matrix and return `(A^{-1}, ln|det(A)|)` via fast
/// closed-form for `n <= 2` or Gauss-Jordan with partial pivoting for `n >= 3`.
fn invert_square_with_logdet(n: usize, mat: &[f64]) -> Result<(Vec<f64>, f64), BackfoldError> {
    if n == 1 {
        let v = mat[0];
        if v.abs() <= 1e-300 || !v.is_finite() {
            return Err(BackfoldError::SingularMatrix("1x1 singular matrix"));
        }
        return Ok((vec![1.0 / v], v.abs().ln()));
    }
    if n == 2 {
        let det = mat[0] * mat[3] - mat[1] * mat[2];
        if det.abs() <= 1e-300 || !det.is_finite() {
            return Err(BackfoldError::SingularMatrix("2x2 singular matrix"));
        }
        let inv = 1.0 / det;
        return Ok((
            vec![mat[3] * inv, -mat[1] * inv, -mat[2] * inv, mat[0] * inv],
            det.abs().ln(),
        ));
    }

    let mut a = mat.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    let mut log_det = 0.0;

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
        log_det += piv_abs.ln();
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
    Ok((inv, log_det))
}

/// Multivariate associative Kalman filter & RTS smoother monoid (`Ctx = usize` state dimension `d`).
///
/// Implements the exact 5-tuple `(A, b, C, eta, J)` + anchor-centered marginal potential `(anchor, log_z)`
/// of Särkkä & García-Fernández (IEEE TAC 2020) with Joseph-form PSD leaves and zero `O(y^2/S)` cancellation.
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
    pub anchor: Vec<f64>,
    pub log_z: f64,
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
        let mut j_mat = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                j_mat[i * dim + j] = fh[i] * fh[j] * inv_s;
            }
        }
        symmetrize(dim, &mut j_mat);

        // Coordinate-aligned anchor satisfying fh^T anchor == y
        let fh_max = fh.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let (anchor, b, eta, log_z) = if fh_max > 1e-150 {
            let idx = fh
                .iter()
                .position(|v| v.abs() > 1e-6 * fh_max)
                .unwrap_or(0);
            let mut anc = vec![0.0; dim];
            anc[idx] = y / fh[idx];
            let b_c = mat_vec(dim, f, &anc);
            (anc, b_c, vec![0.0; dim], -0.5 * (LN_TAU + s.ln()))
        } else {
            let b_c: Vec<f64> = k.iter().map(|&ki| ki * y).collect();
            (
                vec![0.0; dim],
                b_c,
                vec![0.0; dim],
                -0.5 * (LN_TAU + s.ln() + y * y * inv_s),
            )
        };
        Ok(Self {
            dim,
            n: 1,
            a,
            b,
            c,
            eta,
            j: j_mat,
            anchor,
            log_z,
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
        let (s_inv, log_det_s) = invert_square_with_logdet(obs_dim, &s)?;

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

        let mut y_sinv_y = 0.0;
        for p in 0..obs_dim {
            y_sinv_y += y[p] * s_inv_y[p];
        }
        let log_z = -0.5 * ((obs_dim as f64) * LN_TAU + log_det_s + y_sinv_y);
        Ok(Self {
            dim,
            n: 1,
            a,
            b,
            c,
            eta,
            j: j_mat,
            anchor: vec![0.0; dim],
            log_z,
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
            anchor: vec![0.0; dim],
            log_z: 0.0,
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
            anchor: vec![0.0; d],
            log_z: 0.0,
        };
        let combined = Self::combine(&prior, self, &d);
        Ok(KalmanPosterior {
            dim: d,
            mean: combined.b,
            cov: combined.c,
        })
    }

    /// Prior-fusion increment `log p(y_{1:T} | N(m0, P0)) - self.log_z` (cancels `self.log_z`
    /// identically when comparing two priors on the same suffix, e.g. changepoint Bayes factors).
    pub fn log_prior_coupling(&self, m0: &[f64], p0: &[f64]) -> Result<f64, BackfoldError> {
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
            return Ok(0.0);
        }
        let mut delta = vec![0.0; d];
        for i in 0..d {
            delta[i] = m0[i] - self.anchor[i];
        }
        let mut m = mat_mul(d, p0, &self.j);
        for i in 0..d {
            m[i * d + i] += 1.0;
        }
        let (inv_m, log_det_m) = invert_square_with_logdet(d, &m)?;
        let mut inv_mt = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                inv_mt[i * d + j] = inv_m[j * d + i];
            }
        }
        let p_eta = mat_vec(d, p0, &self.eta);
        let j_delta = mat_vec(d, &self.j, &delta);
        let mut sum_eta_rhs = vec![0.0; d];
        for i in 0..d {
            sum_eta_rhs[i] = 2.0 * self.eta[i] - j_delta[i];
        }
        let dt_v = mat_vec(d, &inv_mt, &sum_eta_rhs);
        let d_p_eta = mat_vec(d, &inv_m, &p_eta);
        let mut quad = 0.0;
        for i in 0..d {
            quad += delta[i] * dt_v[i] + self.eta[i] * d_p_eta[i];
        }
        Ok(-0.5 * log_det_m + 0.5 * quad)
    }

    /// Exact Bayesian marginal log-likelihood `log p(y_{1:T})` under initial prior `N(m0, P0)`.
    pub fn log_marginal_likelihood(&self, m0: &[f64], p0: &[f64]) -> Result<f64, BackfoldError> {
        let coupling = self.log_prior_coupling(m0, p0)?;
        if self.n == 0 {
            Ok(0.0)
        } else {
            Ok(self.log_z + coupling)
        }
    }

    /// Exact Rauch-Tung-Striebel (RTS) Bayesian smoothing fusion combining filtered posterior
    /// `p(x_t | y_{1:t}) = N(m_filt, p_filt)` with the future suffix monoid `future` (`y_{t+1:T}`).
    pub fn smooth_fusion(
        m_filt: &[f64],
        p_filt: &[f64],
        future: &Self,
    ) -> Result<KalmanPosterior, BackfoldError> {
        let d = future.dim;
        if m_filt.len() != d || p_filt.len() != d * d {
            return Err(BackfoldError::DimensionMismatch {
                expected: d,
                got: m_filt.len(),
            });
        }
        check_all_finite(m_filt, "m_filt")?;
        check_all_finite(p_filt, "p_filt")?;
        if future.n == 0 {
            return Ok(KalmanPosterior {
                dim: d,
                mean: m_filt.to_vec(),
                cov: p_filt.to_vec(),
            });
        }
        let mut m = mat_mul(d, p_filt, &future.j);
        for i in 0..d {
            m[i * d + i] += 1.0;
        }
        let (inv_m, _) = invert_square_with_logdet(d, &m)?;
        let mut diff = vec![0.0; d];
        for i in 0..d {
            diff[i] = future.anchor[i] - m_filt[i];
        }
        let j_diff = mat_vec(d, &future.j, &diff);
        let mut rhs_inner = vec![0.0; d];
        for i in 0..d {
            rhs_inner[i] = future.eta[i] + j_diff[i];
        }
        let p_rhs = mat_vec(d, p_filt, &rhs_inner);
        let corr = mat_vec(d, &inv_m, &p_rhs);
        let mut mean = vec![0.0; d];
        for i in 0..d {
            mean[i] = m_filt[i] + corr[i];
        }
        let mut cov = mat_mul(d, &inv_m, p_filt);
        symmetrize(d, &mut cov);
        Ok(KalmanPosterior { dim: d, mean, cov })
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
            anchor: vec![0.0; d],
            log_z: 0.0,
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

        let mut delta = vec![0.0; d];
        for i in 0..d {
            delta[i] = e1.b[i] - e2.anchor[i];
        }

        // M = I + C1 * J2 (all eigenvalues >= 1 since C1 >= 0, J2 >= 0)
        let mut m = mat_mul(d, &e1.c, &e2.j);
        for i in 0..d {
            m[i * d + i] += 1.0;
        }
        let (inv_m, log_det_m) = invert_square_with_logdet(d, &m)
            .expect("I + C1*J2 is strictly nonsingular for PSD C1, J2");

        // D^T = (I + J2 * C1)^{-1} = inv_m^T
        let mut inv_mt = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                inv_mt[i * d + j] = inv_m[j * d + i];
            }
        }

        // a2_d = A2 * D
        let a2_d = mat_mul(d, &e2.a, &inv_m);
        let mut a12 = mat_mul(d, &a2_d, &e1.a);

        // b12 = A2 * D * (delta + C1 * eta2) + b2
        let c1_eta2 = mat_vec(d, &e1.c, &e2.eta);
        let mut rhs_b = vec![0.0; d];
        for i in 0..d {
            rhs_b[i] = delta[i] + c1_eta2[i];
        }
        let mut b12 = mat_vec(d, &a2_d, &rhs_b);
        for i in 0..d {
            b12[i] = flush_subnormal(b12[i] + e2.b[i]);
        }

        // C12 = sym(A2 * D * C1 * A2^T + C2)
        let a2_d_c1 = mat_mul(d, &a2_d, &e1.c);
        let mut c12 = mat_mul_t_right(d, &a2_d_c1, &e2.a);
        for i in 0..(d * d) {
            c12[i] = flush_subnormal(c12[i] + e2.c[i]);
        }
        symmetrize(d, &mut c12);

        // a1t_dt = A1^T * D^T
        let a1t_dt = mat_mul_t_left(d, &e1.a, &inv_mt);

        // eta12 = A1^T * D^T * (eta2 - J2 * delta) + eta1
        let j2_delta = mat_vec(d, &e2.j, &delta);
        let mut rhs_e = vec![0.0; d];
        for i in 0..d {
            rhs_e[i] = e2.eta[i] - j2_delta[i];
        }
        let mut eta12 = mat_vec(d, &a1t_dt, &rhs_e);
        for i in 0..d {
            eta12[i] = flush_subnormal(eta12[i] + e1.eta[i]);
        }

        // J12 = sym(A1^T * D^T * J2 * A1 + J1)
        let a1t_dt_j2 = mat_mul(d, &a1t_dt, &e2.j);
        let mut j12 = mat_mul(d, &a1t_dt_j2, &e1.a);
        for i in 0..(d * d) {
            j12[i] = flush_subnormal(j12[i] + e1.j[i]);
        }
        symmetrize(d, &mut j12);
        for v in &mut a12 {
            *v = flush_subnormal(*v);
        }

        // Exact anchor-centered associative marginal log-likelihood update
        let mut sum_eta_rhs = vec![0.0; d];
        for i in 0..d {
            sum_eta_rhs[i] = e2.eta[i] + rhs_e[i];
        }
        let dt_v = mat_vec(d, &inv_mt, &sum_eta_rhs);
        let d_c1_eta2 = mat_vec(d, &inv_m, &c1_eta2);
        let mut quad = 0.0;
        for i in 0..d {
            quad += delta[i] * dt_v[i] + e2.eta[i] * d_c1_eta2[i];
        }
        let log_z = e1.log_z + e2.log_z - 0.5 * log_det_m + 0.5 * quad;

        Self {
            dim: d,
            n: e1.n + e2.n,
            a: a12,
            b: b12,
            c: c12,
            eta: eta12,
            j: j12,
            anchor: e1.anchor.clone(),
            log_z,
        }
    }
}

/// Exact O(log T) Rauch-Tung-Striebel (RTS) smoothed posterior `p(x_t | y_{1:T})` at any
/// timestamp `t` in a persistent `FoldTree<KalmanFilter1D, B>` via a single root-to-leaf walk.
pub fn smooth_1d<const B: usize>(
    tree: &FoldTree<KalmanFilter1D, B>,
    t: i64,
    m0: f64,
    p0: f64,
) -> Result<(f64, f64), BackfoldError> {
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    let (m_f, p_f) = left.posterior(m0, p0)?;
    KalmanFilter1D::smooth_fusion(m_f, p_f, &right)
}

/// Exact O(log T) Rauch-Tung-Striebel (RTS) smoothed posterior `p(x_t | y_{1:T})` at any
/// timestamp `t` in a persistent `FoldTree<KalmanFilter2D, B>` via a single root-to-leaf walk.
pub fn smooth_2d<const B: usize>(
    tree: &FoldTree<KalmanFilter2D, B>,
    t: i64,
    m0: [f64; 2],
    p0: [f64; 4],
) -> Result<([f64; 2], [f64; 4]), BackfoldError> {
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    let (m_f, p_f) = left.posterior(m0, p0)?;
    KalmanFilter2D::smooth_fusion(m_f, p_f, &right)
}

/// Exact O(log T) Rauch-Tung-Striebel (RTS) smoothed posterior `p(x_t | y_{1:T})` at any
/// timestamp `t` in a persistent general-dimension `FoldTree<KalmanMonoid, B>` via a single walk.
pub fn smooth_nd<const B: usize>(
    tree: &FoldTree<KalmanMonoid, B>,
    t: i64,
    m0: &[f64],
    p0: &[f64],
) -> Result<KalmanPosterior, BackfoldError> {
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    let filt = left.posterior(m0, p0)?;
    KalmanMonoid::smooth_fusion(&filt.mean, &filt.cov, &right)
}

/// Exact O(log T) Bayesian changepoint log-Bayes-factor
/// `log(p(y_{<=t}) * p(y_{>t}) / p(y_{1:T}))` testing a full state reset to `N(m0, p0)` right
/// after timestamp `t` (1D). Uses exact prior-coupling differencing so `right.log_z` cancels
/// algebraically with zero catastrophic cancellation at arbitrary horizon `T`.
pub fn changepoint_bayes_factor_1d<const B: usize>(
    tree: &FoldTree<KalmanFilter1D, B>,
    t: i64,
    m0: f64,
    p0: f64,
) -> Result<f64, BackfoldError> {
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    if left.n == 0 || right.n == 0 {
        return Ok(0.0);
    }
    let (mf, pf) = left.posterior(m0, p0)?;
    let c_reset = right.log_prior_coupling(m0, p0)?;
    let c_cont = right.log_prior_coupling(mf, pf)?;
    Ok(c_reset - c_cont)
}

/// Exact O(log T) translation-invariant Bayesian state-shock log-Bayes-factor testing an
/// instantaneous shock `x_t^+ ~ N(m_f, p_f + q_jump)` vs continuity `N(m_f, p_f)` after `t` (1D).
pub fn changepoint_shock_bayes_factor_1d<const B: usize>(
    tree: &FoldTree<KalmanFilter1D, B>,
    t: i64,
    m0: f64,
    p0: f64,
    q_jump: f64,
) -> Result<f64, BackfoldError> {
    let q_jump = check_finite(q_jump, "q_jump")?;
    if q_jump < 0.0 {
        return Err(BackfoldError::InvalidParameter("q_jump must be >= 0"));
    }
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    if left.n == 0 || right.n == 0 {
        return Ok(0.0);
    }
    let (mf, pf) = left.posterior(m0, p0)?;
    let c_jump = right.log_prior_coupling(mf, pf + q_jump)?;
    let c_cont = right.log_prior_coupling(mf, pf)?;
    Ok(c_jump - c_cont)
}

/// Exact O(log T) Bayesian changepoint log-Bayes-factor
/// `log(p(y_{<=t}) * p(y_{>t}) / p(y_{1:T}))` testing a state reset to `N(m0, P0)` after `t` (2D).
pub fn changepoint_bayes_factor_2d<const B: usize>(
    tree: &FoldTree<KalmanFilter2D, B>,
    t: i64,
    m0: [f64; 2],
    p0: [f64; 4],
) -> Result<f64, BackfoldError> {
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    if left.n == 0 || right.n == 0 {
        return Ok(0.0);
    }
    let (mf, pf) = left.posterior(m0, p0)?;
    let c_reset = right.log_prior_coupling(m0, p0)?;
    let c_cont = right.log_prior_coupling(mf, pf)?;
    Ok(c_reset - c_cont)
}

/// Exact O(log T) translation-invariant Bayesian state-shock log-Bayes-factor testing an
/// instantaneous shock `x_t^+ ~ N(m_f, P_f + Q_jump)` vs continuity `N(m_f, P_f)` after `t` (2D).
pub fn changepoint_shock_bayes_factor_2d<const B: usize>(
    tree: &FoldTree<KalmanFilter2D, B>,
    t: i64,
    m0: [f64; 2],
    p0: [f64; 4],
    q_jump: [f64; 4],
) -> Result<f64, BackfoldError> {
    check_all_finite(&q_jump, "q_jump")?;
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    if left.n == 0 || right.n == 0 {
        return Ok(0.0);
    }
    let (mf, pf) = left.posterior(m0, p0)?;
    let p_jump = sym2([
        pf[0] + q_jump[0],
        pf[1] + q_jump[1],
        pf[2] + q_jump[2],
        pf[3] + q_jump[3],
    ]);
    let c_jump = right.log_prior_coupling(mf, p_jump)?;
    let c_cont = right.log_prior_coupling(mf, pf)?;
    Ok(c_jump - c_cont)
}

/// Exact O(log T) Bayesian changepoint log-Bayes-factor
/// `log(p(y_{<=t}) * p(y_{>t}) / p(y_{1:T}))` testing a state reset to `N(m0, P0)` after `t` (ND).
pub fn changepoint_bayes_factor_nd<const B: usize>(
    tree: &FoldTree<KalmanMonoid, B>,
    t: i64,
    m0: &[f64],
    p0: &[f64],
) -> Result<f64, BackfoldError> {
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    if left.n == 0 || right.n == 0 {
        return Ok(0.0);
    }
    let filt = left.posterior(m0, p0)?;
    let c_reset = right.log_prior_coupling(m0, p0)?;
    let c_cont = right.log_prior_coupling(&filt.mean, &filt.cov)?;
    Ok(c_reset - c_cont)
}

/// Exact O(log T) translation-invariant Bayesian state-shock log-Bayes-factor testing an
/// instantaneous shock `x_t^+ ~ N(m_f, P_f + Q_jump)` vs continuity `N(m_f, P_f)` after `t` (ND).
pub fn changepoint_shock_bayes_factor_nd<const B: usize>(
    tree: &FoldTree<KalmanMonoid, B>,
    t: i64,
    m0: &[f64],
    p0: &[f64],
    q_jump: &[f64],
) -> Result<f64, BackfoldError> {
    check_all_finite(q_jump, "q_jump")?;
    let (left, right) = tree.split_summary(EventKey::time_end(t));
    if left.n == 0 || right.n == 0 {
        return Ok(0.0);
    }
    let filt = left.posterior(m0, p0)?;
    if q_jump.len() != filt.cov.len() {
        return Err(BackfoldError::DimensionMismatch {
            expected: filt.cov.len(),
            got: q_jump.len(),
        });
    }
    let mut p_jump = vec![0.0; filt.cov.len()];
    for i in 0..filt.cov.len() {
        p_jump[i] = filt.cov[i] + q_jump[i];
    }
    symmetrize(filt.dim, &mut p_jump);
    let c_jump = right.log_prior_coupling(&filt.mean, &p_jump)?;
    let c_cont = right.log_prior_coupling(&filt.mean, &filt.cov)?;
    Ok(c_jump - c_cont)
}

/// Zero-blind-spot hierarchical Bayesian shock changepoint locator (1D).
///
/// Reads cached subtree `log_z` summaries directly from `FoldTree` (zero `combine` calls during
/// screening, checking both `<= 64`-tick subtrees and their `<= 256`-tick parents so no boundary
/// shock can fall in a crack), zooms the top-3 highest excess-NLL regions, and evaluates exact
/// single-tick `changepoint_shock_bayes_factor_1d`.
pub fn locate_changepoint_1d<const B: usize>(
    tree: &FoldTree<KalmanFilter1D, B>,
    m0: f64,
    p0: f64,
    q_jump: f64,
    n_windows: usize,
) -> Result<Option<(i64, f64)>, BackfoldError> {
    let n = tree.len();
    if n < 4 {
        return Ok(None);
    }
    let baseline = tree.summary().log_z / (n as f64);
    let max_sub_len = (n / n_windows.clamp(8, 256)).clamp(16, 64);
    let mut cands = tree.scan_cached_subtrees(max_sub_len, |m, len| {
        if len < 2 {
            f64::NEG_INFINITY
        } else {
            baseline * (len as f64) - m.log_z
        }
    });
    cands.sort_by(|a, b| b.3.total_cmp(&a.3));

    let score_win = |a: i64, b: i64| -> (u64, f64) {
        let seg = tree.query_time_range(a, b);
        if seg.n < 2 {
            (seg.n, f64::NEG_INFINITY)
        } else {
            (seg.n, baseline * (seg.n as f64) - seg.log_z)
        }
    };

    let mut best: Option<(i64, f64)> = None;
    for &(k_lo, k_hi, _, _) in cands.iter().take(4) {
        let mut lo = k_lo.event_time;
        let mut hi = k_hi.event_time;
        for _ in 0..16 {
            let (cnt, _) = score_win(lo, hi);
            if cnt <= 24 || hi - lo <= 4 {
                break;
            }
            let mid = lo + (hi - lo) / 2;
            let ov = ((hi - lo) / 8).clamp(2, 32);
            let l_hi = (mid + ov).min(hi);
            let r_lo = (mid - ov).max(lo);
            let (_, ls) = score_win(lo, l_hi);
            let (_, rs) = score_win(r_lo, hi);
            if ls >= rs {
                if l_hi == hi {
                    break;
                }
                hi = l_hi;
            } else {
                if r_lo == lo {
                    break;
                }
                lo = r_lo;
            }
        }
        let mut last_t = i64::MIN;
        for (k, _) in tree.iter_range(EventKey::time_start(lo), EventKey::time_end(hi)) {
            if k.event_time == last_t {
                continue;
            }
            last_t = k.event_time;
            let bf = changepoint_shock_bayes_factor_1d(tree, k.event_time, m0, p0, q_jump)?;
            if best.is_none_or(|(_, b)| bf > b) {
                best = Some((k.event_time, bf));
            }
        }
    }
    Ok(best)
}

/// Zero-blind-spot hierarchical Bayesian shock changepoint locator (2D).
///
/// Screens cached `FoldTree` subtrees (`<= 64` ticks and `<= 256`-tick boundary-covering parents)
/// with zero `combine` overhead, zooms top-4 excess-NLL regions, and evaluates exact single-tick
/// `changepoint_shock_bayes_factor_2d` across every tick in the zoomed intervals.
pub fn locate_changepoint_2d<const B: usize>(
    tree: &FoldTree<KalmanFilter2D, B>,
    m0: [f64; 2],
    p0: [f64; 4],
    q_jump: [f64; 4],
    n_windows: usize,
) -> Result<Option<(i64, f64)>, BackfoldError> {
    let n = tree.len();
    if n < 4 {
        return Ok(None);
    }
    let baseline = tree.summary().log_z / (n as f64);
    let max_sub_len = (n / n_windows.clamp(8, 256)).clamp(16, 64);
    let mut cands = tree.scan_cached_subtrees(max_sub_len, |m, len| {
        if len < 2 {
            f64::NEG_INFINITY
        } else {
            baseline * (len as f64) - m.log_z
        }
    });
    cands.sort_by(|a, b| b.3.total_cmp(&a.3));

    let score_win = |a: i64, b: i64| -> (u64, f64) {
        let seg = tree.query_time_range(a, b);
        if seg.n < 2 {
            (seg.n, f64::NEG_INFINITY)
        } else {
            (seg.n, baseline * (seg.n as f64) - seg.log_z)
        }
    };

    let mut best: Option<(i64, f64)> = None;
    for &(k_lo, k_hi, _, _) in cands.iter().take(4) {
        let mut lo = k_lo.event_time;
        let mut hi = k_hi.event_time;
        for _ in 0..16 {
            let (cnt, _) = score_win(lo, hi);
            if cnt <= 24 || hi - lo <= 4 {
                break;
            }
            let mid = lo + (hi - lo) / 2;
            let ov = ((hi - lo) / 8).clamp(2, 32);
            let l_hi = (mid + ov).min(hi);
            let r_lo = (mid - ov).max(lo);
            let (_, ls) = score_win(lo, l_hi);
            let (_, rs) = score_win(r_lo, hi);
            if ls >= rs {
                if l_hi == hi {
                    break;
                }
                hi = l_hi;
            } else {
                if r_lo == lo {
                    break;
                }
                lo = r_lo;
            }
        }
        let mut last_t = i64::MIN;
        for (k, _) in tree.iter_range(EventKey::time_start(lo), EventKey::time_end(hi)) {
            if k.event_time == last_t {
                continue;
            }
            last_t = k.event_time;
            let bf = changepoint_shock_bayes_factor_2d(tree, k.event_time, m0, p0, q_jump)?;
            if best.is_none_or(|(_, b)| bf > b) {
                best = Some((k.event_time, bf));
            }
        }
    }
    Ok(best)
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

    #[test]
    fn rts_smoother_and_marginal_log_likelihood_exact_parity() {
        // Irregular dt + missing observations + full textbook RTS forward-backward reference
        let dts = [1.0f64, 0.5, 2.0, 0.25, 1.5, 1.0, 0.75, 1.25, 0.5, 1.0];
        let ys = [
            Some(1.1),
            Some(1.8),
            None,
            Some(4.2),
            Some(5.9),
            None,
            Some(7.7),
            Some(9.1),
            Some(9.5),
            Some(10.6),
        ];
        let q_acc = 0.08;
        let r_obs = 0.35;
        let m0 = [0.4, 0.3];
        let p0 = [6.0, 0.15, 0.15, 2.5];

        let mut tree2: FoldTree<KalmanFilter2D, 4> = FoldTree::new(());
        let mut treen: FoldTree<KalmanMonoid, 4> = FoldTree::new(2);

        // Forward filter storing (m_filt, p_filt, m_pred, p_pred, F) for textbook RTS backward pass
        let n = dts.len();
        let mut m_filt = vec![[0.0; 2]; n + 1];
        let mut p_filt = vec![[0.0; 4]; n + 1];
        let mut m_pred = vec![[0.0; 2]; n];
        let mut p_pred = vec![[0.0; 4]; n];
        let mut fs = vec![[0.0; 4]; n];
        m_filt[0] = m0;
        p_filt[0] = p0;
        let mut seq_ll = 0.0;

        for t in 0..n {
            let dt = dts[t];
            let f = [1.0, dt, 0.0, 1.0];
            let q = [
                q_acc * dt.powi(3) / 3.0,
                q_acc * dt.powi(2) / 2.0,
                q_acc * dt.powi(2) / 2.0,
                q_acc * dt,
            ];
            fs[t] = f;
            let mp = mv2(&f, &m_filt[t]);
            let fp = mul2(&f, &p_filt[t]);
            let fpf = mul2_tr(&fp, &f);
            let pp = sym2([fpf[0] + q[0], fpf[1] + q[1], fpf[2] + q[2], fpf[3] + q[3]]);
            m_pred[t] = mp;
            p_pred[t] = pp;

            let ts = (t + 1) as i64;
            if let Some(y) = ys[t] {
                tree2.upsert(
                    EventKey::at(ts),
                    KalmanFilter2D::from_constant_velocity(dt, q_acc, r_obs, y).unwrap(),
                );
                treen.upsert(
                    EventKey::at(ts),
                    KalmanMonoid::from_constant_velocity(dt, q_acc, r_obs, y).unwrap(),
                );
                let s = pp[0] + r_obs;
                let innov = y - mp[0];
                seq_ll += -0.5 * (LN_TAU + s.ln() + innov * innov / s);
                let k = [pp[0] / s, pp[2] / s];
                m_filt[t + 1] = [mp[0] + k[0] * innov, mp[1] + k[1] * innov];
                let ikh = [1.0 - k[0], 0.0, -k[1], 1.0];
                let tmp = mul2(&ikh, &pp);
                let jcov = mul2_tr(&tmp, &ikh);
                p_filt[t + 1] = sym2([
                    jcov[0] + k[0] * k[0] * r_obs,
                    jcov[1] + k[0] * k[1] * r_obs,
                    jcov[2] + k[1] * k[0] * r_obs,
                    jcov[3] + k[1] * k[1] * r_obs,
                ]);
            } else {
                tree2.upsert(EventKey::at(ts), KalmanFilter2D::from_missing(f, q).unwrap());
                treen.upsert(
                    EventKey::at(ts),
                    KalmanMonoid::from_missing(2, &f, &q).unwrap(),
                );
                m_filt[t + 1] = mp;
                p_filt[t + 1] = pp;
            }
        }

        // Exact marginal log-likelihood parity (both 2D fast path and general KalmanMonoid!)
        let ll2 = tree2.summary().log_marginal_likelihood(m0, p0).unwrap();
        let lln = treen.summary().log_marginal_likelihood(&m0, &p0).unwrap();
        assert!((ll2 - seq_ll).abs() < 1e-11, "2D ll={ll2} vs seq={seq_ll}");
        assert!((lln - seq_ll).abs() < 1e-11, "ND ll={lln} vs seq={seq_ll}");

        // Textbook Rauch-Tung-Striebel backward pass for ALL states t = 0..=n
        let mut m_rts = m_filt.clone();
        let mut p_rts = p_filt.clone();
        for t in (0..n).rev() {
            let pp = p_pred[t];
            let det = pp[0] * pp[3] - pp[1] * pp[2];
            let inv_pp = [pp[3] / det, -pp[1] / det, -pp[2] / det, pp[0] / det];
            // G_t = P_{t|t} F_t^T P_{t+1|t}^{-1}
            let pf_t = mul2_tr(&p_filt[t], &fs[t]);
            let g = mul2(&pf_t, &inv_pp);
            let dm = [m_rts[t + 1][0] - m_pred[t][0], m_rts[t + 1][1] - m_pred[t][1]];
            let g_dm = mv2(&g, &dm);
            m_rts[t] = [m_filt[t][0] + g_dm[0], m_filt[t][1] + g_dm[1]];
            let dp = [
                p_rts[t + 1][0] - pp[0],
                p_rts[t + 1][1] - pp[1],
                p_rts[t + 1][2] - pp[2],
                p_rts[t + 1][3] - pp[3],
            ];
            let g_dp = mul2(&g, &dp);
            let g_dp_gt = mul2_tr(&g_dp, &g);
            p_rts[t] = sym2([
                p_filt[t][0] + g_dp_gt[0],
                p_filt[t][1] + g_dp_gt[1],
                p_filt[t][2] + g_dp_gt[2],
                p_filt[t][3] + g_dp_gt[3],
            ]);
        }

        // Verify smooth_2d and smooth_nd match textbook RTS at EVERY timestamp 0..=n!
        for t in 0..=n {
            let (ms2, ps2) = smooth_2d(&tree2, t as i64, m0, p0).unwrap();
            let postn = smooth_nd(&treen, t as i64, &m0, &p0).unwrap();
            for i in 0..2 {
                assert!(
                    (ms2[i] - m_rts[t][i]).abs() < 1e-11,
                    "smooth_2d mean mismatch at t={t}[{i}]: {} vs {}",
                    ms2[i],
                    m_rts[t][i]
                );
                assert!(
                    (postn.mean[i] - m_rts[t][i]).abs() < 1e-11,
                    "smooth_nd mean mismatch at t={t}[{i}]"
                );
            }
            for i in 0..4 {
                assert!(
                    (ps2[i] - p_rts[t][i]).abs() < 1e-11,
                    "smooth_2d cov mismatch at t={t}[{i}]: {} vs {}",
                    ps2[i],
                    p_rts[t][i]
                );
                assert!(
                    (postn.cov[i] - p_rts[t][i]).abs() < 1e-11,
                    "smooth_nd cov mismatch at t={t}[{i}]"
                );
            }
        }
    }

    #[test]
    fn changepoint_bayes_factor_exact_parity_and_shock_localization() {
        let n = 250usize;
        let true_cp = 120usize;
        let offset = 1e6_f64;
        let mut entries1 = Vec::with_capacity(n);
        let mut entries2 = Vec::with_capacity(n);
        let mut entries_n = Vec::with_capacity(n);
        let mut seed = 987654321u64;
        for t in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5) * 0.4;
            let y = if t <= true_cp {
                0.05 * (t as f64) + noise
            } else {
                0.05 * (t as f64) + 12.0 + 0.15 * ((t - true_cp) as f64) + noise
            };
            entries1.push((
                EventKey::at(t as i64),
                KalmanFilter1D::from_random_walk(1.0, 0.05, 0.25, y).unwrap(),
            ));
            entries2.push((
                EventKey::at(t as i64),
                KalmanFilter2D::from_constant_velocity(1.0, 1e-3, 0.25, y + offset).unwrap(),
            ));
            entries_n.push((
                EventKey::at(t as i64),
                KalmanMonoid::from_constant_velocity(1.0, 1e-3, 0.25, y).unwrap(),
            ));
        }
        let tree1: FoldTree<KalmanFilter1D, 8> = FoldTree::from_sorted((), entries1);
        let tree2: FoldTree<KalmanFilter2D, 2> = FoldTree::from_sorted((), entries2);
        let treen: FoldTree<KalmanMonoid, 2> = FoldTree::from_sorted(2, entries_n);

        // 1. Exact parity with brute-force (ll_left + ll_right - ll_full) for 1D, 2D, ND
        let m0_2 = [0.0, 0.0];
        let p0_2 = [100.0, 0.0, 0.0, 10.0];
        for &split_t in &[30i64, 120, 180] {
            let bf1 = changepoint_bayes_factor_1d(&tree1, split_t, 0.0, 100.0).unwrap();
            let (l1, r1) = tree1.split_summary(EventKey::time_end(split_t));
            let ref1 = l1.log_marginal_likelihood(0.0, 100.0).unwrap()
                + r1.log_marginal_likelihood(0.0, 100.0).unwrap()
                - tree1.summary().log_marginal_likelihood(0.0, 100.0).unwrap();
            assert!((bf1 - ref1).abs() < 1e-10, "1D BF mismatch: {bf1} vs {ref1}");

            let bfn = changepoint_bayes_factor_nd(&treen, split_t, &m0_2, &p0_2).unwrap();
            let (ln, rn) = treen.split_summary(EventKey::time_end(split_t));
            let refn = ln.log_marginal_likelihood(&m0_2, &p0_2).unwrap()
                + rn.log_marginal_likelihood(&m0_2, &p0_2).unwrap()
                - treen.summary().log_marginal_likelihood(&m0_2, &p0_2).unwrap();
            assert!((bfn - refn).abs() < 1e-10, "ND BF mismatch: {bfn} vs {refn}");
        }

        // 2. Translation-invariant shock Bayes factor at +1e6 offset localizes exact changepoint!
        let m0_hi = [offset, 0.0];
        let q_jump = [200.0, 0.0, 0.0, 1.0];
        let mut best_t = 0i64;
        let mut best_bf = f64::NEG_INFINITY;
        for t in 10..240i64 {
            let bf = changepoint_shock_bayes_factor_2d(&tree2, t, m0_hi, p0_2, q_jump).unwrap();
            if bf > best_bf {
                best_bf = bf;
                best_t = t;
            }
        }
        assert_eq!(best_t, true_cp as i64, "expected peak at {true_cp}, got {best_t} (bf={best_bf})");
        assert!(best_bf > 40.0, "expected huge log-BF at true shock, got {best_bf}");
        let calm_bf = changepoint_shock_bayes_factor_2d(&tree2, 40, m0_hi, p0_2, q_jump).unwrap();
        assert!(calm_bf < 0.0, "calm region should have negative Occam log-BF, got {calm_bf}");
    }
}
