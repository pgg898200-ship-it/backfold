//! Zero-copy PyO3 Python bindings for `backfold`.
//!
//! Exposes persistent O(log T) out-of-order, retractable, forkable streaming operators:
//! - `MomentsStream` (compensated Chan-Pébay-Terriberry moments)
//! - `EwmaStream` (continuous-time irregular-dt EWMA)
//! - `CusumStream` (two-sided tropical-semiring Page CUSUM + O(log T) first-crossing search)
//! - `EProcessStream` (anytime-valid betting martingale + Page e-CUSUM + Shiryaev-Roberts e-value)
//! - `KalmanStream` (zero-alloc 1D/2D + general-dim Särkkä-García-Fernández filter, exact O(log T) RTS smoother & marginal log-likelihood)
//! - `CovarianceStream` (106-bit compensated multivariate covariance, correlation & Mahalanobis anomaly monoid)
//! - `RidgeStream` (continuous-time discounted Ridge with Jacobi-equilibrated Cholesky)
//! - `SwarmEProcessLedger` (echo-proof conflict-free multi-agent CRDT ledger)
//! - `BitemporalMoments` (zero-lookahead point-in-time bitemporal query engine)
//! - `ebh_reject`, `digest_stream`, `causal_robust_scale`

#![allow(clippy::too_many_arguments)]

use backfold_core::digest::{build_stream_digest, causal_robust_scale as core_robust_scale, StreamDigest};
use backfold_core::ops::{
    changepoint_bayes_factor_1d, changepoint_bayes_factor_2d, changepoint_bayes_factor_nd,
    changepoint_shock_bayes_factor_1d, changepoint_shock_bayes_factor_2d,
    changepoint_shock_bayes_factor_nd, ebh_reject as core_ebh_reject, locate_changepoint_1d,
    locate_changepoint_2d, sift_streams_fdr as core_sift_streams_fdr, smooth_1d, smooth_2d,
    smooth_nd, CovarianceMonoid,
    CovariancePca, CusumMonoid, EProcessMonoid, EwmaMonoid, KalmanFilter1D, KalmanFilter2D,
    KalmanMonoid, MomentsMonoid, RidgeMonoid, RidgeSolution,
};
use backfold_core::swarm::{BitemporalFold, SwarmLedger};
use backfold_core::{BackfoldError, EventKey, FoldTree};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

type SmoothBatch = (Vec<Vec<f64>>, Vec<Vec<f64>>);

fn to_py_err(e: BackfoldError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Extract a 1D `i64` sequence via zero-copy PEP-3118 buffer (NumPy) or Python sequence fallback.
fn extract_i64_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    if let Ok(buf) = PyBuffer::<i64>::get(obj) {
        if let Some(slice) = buf.as_slice(py) {
            return Ok(slice.iter().map(|c| c.get()).collect());
        }
    }
    obj.extract::<Vec<i64>>()
}

/// Extract a 1D `f64` sequence via zero-copy PEP-3118 buffer (NumPy) or Python sequence fallback.
fn extract_f64_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        if let Some(slice) = buf.as_slice(py) {
            return Ok(slice.iter().map(|c| c.get()).collect());
        }
    }
    obj.extract::<Vec<f64>>()
}

/// Extract a 2D row-major `f64` matrix (`n x dim`) via zero-copy NumPy buffer or nested list fallback.
fn extract_f64_mat(py: Python<'_>, obj: &Bound<'_, PyAny>, expected_dim: usize) -> PyResult<(Vec<f64>, usize)> {
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        let shape = buf.shape();
        if shape.len() == 2 {
            let n = shape[0];
            let d = shape[1];
            if d != expected_dim {
                return Err(PyValueError::new_err(format!(
                    "dimension mismatch: expected {expected_dim} columns, got {d}"
                )));
            }
            if let Some(slice) = buf.as_slice(py) {
                return Ok((slice.iter().map(|c| c.get()).collect(), n));
            }
        }
    }
    let rows: Vec<Vec<f64>> = obj.extract()?;
    let n = rows.len();
    let mut flat = Vec::with_capacity(n * expected_dim);
    for row in rows {
        if row.len() != expected_dim {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: expected {expected_dim}, got {}",
                row.len()
            )));
        }
        flat.extend_from_slice(&row);
    }
    Ok((flat, n))
}

/// Extract an arbitrary-shaped 2D row-major `f64` matrix (`n_rows x n_cols`).
fn extract_f64_2d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        let shape = buf.shape();
        if shape.len() == 2 {
            if let Some(slice) = buf.as_slice(py) {
                return Ok((slice.iter().map(|c| c.get()).collect(), shape[0], shape[1]));
            }
        }
    }
    let rows: Vec<Vec<f64>> = obj.extract()?;
    let n_rows = rows.len();
    let n_cols = rows.first().map_or(0, Vec::len);
    let mut flat = Vec::with_capacity(n_rows * n_cols);
    for row in rows {
        if row.len() != n_cols {
            return Err(PyValueError::new_err(format!(
                "ragged 2D input: expected {n_cols} columns, got {}",
                row.len()
            )));
        }
        flat.extend_from_slice(&row);
    }
    Ok((flat, n_rows, n_cols))
}

fn pca_to_dict<'py>(py: Python<'py>, pca: &CovariancePca) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("dim", pca.dim)?;
    d.set_item("k", pca.k)?;
    d.set_item("eigenvalues", &pca.eigenvalues)?;
    d.set_item("explained_variance_ratio", &pca.explained_variance_ratio)?;
    let comp = flat_to_rows(&pca.components, pca.dim);
    d.set_item("components", comp)?;
    Ok(d)
}

fn moments_to_dict<'py>(py: Python<'py>, m: &MomentsMonoid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", m.n)?;
    d.set_item("mean", m.mean)?;
    d.set_item("variance", m.variance())?;
    d.set_item("sample_variance", m.sample_variance())?;
    d.set_item("std", m.std())?;
    d.set_item("sample_std", m.sample_std())?;
    d.set_item("skewness", m.skewness())?;
    d.set_item("excess_kurtosis", m.excess_kurtosis())?;
    d.set_item("min", if m.n > 0 { Some(m.min) } else { None })?;
    d.set_item("max", if m.n > 0 { Some(m.max) } else { None })?;
    d.set_item("t_min", if m.n > 0 { Some(m.t_min) } else { None })?;
    d.set_item("t_max", if m.n > 0 { Some(m.t_max) } else { None })?;
    d.set_item("t_first", if m.n > 0 { Some(m.t_first) } else { None })?;
    d.set_item("t_last", if m.n > 0 { Some(m.t_last) } else { None })?;
    Ok(d)
}

fn ewma_to_dict<'py>(py: Python<'py>, m: &EwmaMonoid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", m.n)?;
    d.set_item("w_sum", m.w_sum)?;
    d.set_item("mean", m.mean)?;
    d.set_item("variance", m.variance())?;
    d.set_item("std", m.std())?;
    d.set_item("t_first", if m.n > 0 { Some(m.t_first) } else { None })?;
    d.set_item("t_last", if m.n > 0 { Some(m.t_last) } else { None })?;
    Ok(d)
}

fn cusum_to_dict<'py>(py: Python<'py>, m: &CusumMonoid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", m.n)?;
    d.set_item("current_pos", m.current_pos())?;
    d.set_item("current_neg", m.current_neg())?;
    d.set_item("peak_pos", m.peak_pos())?;
    d.set_item("peak_neg", m.peak_neg())?;
    d.set_item("max_peak", m.max_peak())?;
    d.set_item(
        "onset_ts",
        if m.onset_ts() != i64::MIN {
            Some(m.onset_ts())
        } else {
            None
        },
    )?;
    d.set_item(
        "peak_ts",
        if m.peak_ts() != i64::MIN {
            Some(m.peak_ts())
        } else {
            None
        },
    )?;
    Ok(d)
}

fn eprocess_to_dict<'py>(py: Python<'py>, m: &EProcessMonoid) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", m.n)?;
    d.set_item("log_wealth", m.log_wealth())?;
    d.set_item("wealth", m.wealth())?;
    d.set_item("log_e_detector", m.log_e_detector())?;
    d.set_item("e_detector", m.e_detector())?;
    d.set_item("sr_evalue", m.sr_evalue())?;
    d.set_item("scan_evalue", m.scan_evalue())?;
    d.set_item("e_value", m.e_value())?;
    d.set_item("peak_e_detector_valid", m.peak_e_detector_valid())?;
    d.set_item("anytime_p_value", m.anytime_p_value())?;
    d.set_item(
        "peak_ts",
        if m.peak_ts != i64::MIN {
            Some(m.peak_ts)
        } else {
            None
        },
    )?;
    Ok(d)
}

fn ridge_sol_to_dict<'py>(py: Python<'py>, s: &RidgeSolution) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("weights", &s.weights)?;
    d.set_item("residual_variance", s.residual_variance)?;
    d.set_item("effective_weight", s.effective_weight)?;
    d.set_item("n", s.n)?;
    d.set_item("condition_est", s.condition_est)?;
    Ok(d)
}

fn digest_to_dict<'py>(py: Python<'py>, dig: &StreamDigest) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("channel", &dig.channel)?;
    d.set_item("n_ticks", dig.n_ticks)?;
    d.set_item("t_first", dig.t_first)?;
    d.set_item("t_last", dig.t_last)?;
    d.set_item("raw_approx_tokens", dig.raw_approx_tokens)?;
    d.set_item("digest_approx_tokens", dig.digest_approx_tokens)?;
    d.set_item("compression_ratio", dig.compression_ratio)?;
    d.set_item("baseline_median", dig.baseline_median)?;
    d.set_item("baseline_mad_std", dig.baseline_mad_std)?;
    d.set_item("global_mean", dig.global_mean)?;
    d.set_item("global_std", dig.global_std)?;
    d.set_item("global_min", dig.global_min)?;
    d.set_item("global_min_ts", dig.global_min_ts)?;
    d.set_item("global_max", dig.global_max)?;
    d.set_item("global_max_ts", dig.global_max_ts)?;
    d.set_item("cusum_peak_pos", dig.cusum_peak_pos)?;
    d.set_item("cusum_peak_neg", dig.cusum_peak_neg)?;
    d.set_item("cusum_onset_ts", dig.cusum_onset_ts)?;
    d.set_item("cusum_peak_ts", dig.cusum_peak_ts)?;
    d.set_item("first_alarm_ts", dig.first_alarm_ts)?;
    d.set_item("e_detector", dig.e_detector)?;
    d.set_item("e_detector_peak_valid", dig.e_detector_peak_valid)?;
    d.set_item("anytime_p_value", dig.anytime_p_value)?;
    d.set_item("kalman_level", dig.kalman_level)?;
    d.set_item("kalman_std", dig.kalman_std)?;

    let hs = PyList::empty(py);
    for w in &dig.hotspots {
        let wd = PyDict::new(py);
        wd.set_item("t_start", w.t_start)?;
        wd.set_item("t_end", w.t_end)?;
        wd.set_item("count", w.count)?;
        wd.set_item("mean", w.mean)?;
        wd.set_item("std", w.std)?;
        wd.set_item("min", w.min)?;
        wd.set_item("max", w.max)?;
        wd.set_item("t_min", w.t_min)?;
        wd.set_item("t_max", w.t_max)?;
        wd.set_item("surprise_z", w.surprise_z)?;
        wd.set_item("kind", &w.kind)?;
        hs.append(wd)?;
    }
    d.set_item("hotspots", hs)?;
    d.set_item("markdown", dig.to_markdown())?;
    Ok(d)
}

#[inline]
fn is_non_decreasing(ts: &[i64]) -> bool {
    ts.windows(2).all(|w| w[0] <= w[1])
}

// ---------------------------------------------------------------------------
// 1. MomentsStream
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct MomentsStream {
    tree: FoldTree<MomentsMonoid, 4>,
}

#[pymethods]
impl MomentsStream {
    #[new]
    fn new() -> Self {
        Self {
            tree: FoldTree::new(()),
        }
    }

    #[pyo3(signature = (ts, xs, source_id = 0))]
    fn ingest(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        xs: &Bound<'_, PyAny>,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let x_vec = extract_f64_vec(py, xs)?;
        if t_vec.len() != x_vec.len() {
            return Err(PyValueError::new_err("ts and xs must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for (&t, &x) in t_vec.iter().zip(x_vec.iter()) {
                let m = MomentsMonoid::singleton(t, x).map_err(to_py_err)?;
                entries.push((EventKey::sourced(t, source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                self.tree = FoldTree::from_sorted((), entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, x, source_id = 0, seq = 0))]
    fn upsert(&mut self, ts: i64, x: f64, source_id: u64, seq: u64) -> PyResult<()> {
        let m = MomentsMonoid::singleton(ts, x).map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        self.tree.retract(EventKey::sourced(ts, source_id, seq)).is_some()
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        moments_to_dict(py, &self.tree.summary())
    }

    fn range<'py>(&self, py: Python<'py>, t_start: i64, t_end: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.range(EventKey::time_start(t_start), EventKey::time_end(t_end));
        moments_to_dict(py, &m)
    }

    fn prefix<'py>(&self, py: Python<'py>, t_end: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.prefix(EventKey::time_end(t_end));
        moments_to_dict(py, &m)
    }

    fn suffix<'py>(&self, py: Python<'py>, t_start: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.suffix(EventKey::time_start(t_start));
        moments_to_dict(py, &m)
    }

    fn fork(&self) -> Self {
        Self {
            tree: self.tree.fork(),
        }
    }

    fn merge(&mut self, other: &MomentsStream) {
        self.tree = self.tree.merge(&other.tree);
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }
}

// ---------------------------------------------------------------------------
// 2. EwmaStream
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct EwmaStream {
    tree: FoldTree<EwmaMonoid, 4>,
}

#[pymethods]
impl EwmaStream {
    #[new]
    #[pyo3(signature = (half_life = None, decay_rate = None))]
    fn new(half_life: Option<f64>, decay_rate: Option<f64>) -> PyResult<Self> {
        let lambda = match (half_life, decay_rate) {
            (Some(hl), None) => EwmaMonoid::decay_rate_from_half_life(hl).map_err(to_py_err)?,
            (None, Some(dr)) => {
                if !dr.is_finite() || dr < 0.0 {
                    return Err(PyValueError::new_err("decay_rate must be finite and >= 0"));
                }
                dr
            }
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "specify either half_life or decay_rate, not both",
                ))
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "must specify either half_life or decay_rate",
                ))
            }
        };
        Ok(Self {
            tree: FoldTree::new(lambda),
        })
    }

    #[pyo3(signature = (ts, xs, source_id = 0))]
    fn ingest(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        xs: &Bound<'_, PyAny>,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let x_vec = extract_f64_vec(py, xs)?;
        if t_vec.len() != x_vec.len() {
            return Err(PyValueError::new_err("ts and xs must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for (&t, &x) in t_vec.iter().zip(x_vec.iter()) {
                let m = EwmaMonoid::singleton(t, x).map_err(to_py_err)?;
                entries.push((EventKey::sourced(t, source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                let ctx = *self.tree.ctx();
                self.tree = FoldTree::from_sorted(ctx, entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, x, source_id = 0, seq = 0))]
    fn upsert(&mut self, ts: i64, x: f64, source_id: u64, seq: u64) -> PyResult<()> {
        let m = EwmaMonoid::singleton(ts, x).map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        self.tree.retract(EventKey::sourced(ts, source_id, seq)).is_some()
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        ewma_to_dict(py, &self.tree.summary())
    }

    fn range<'py>(&self, py: Python<'py>, t_start: i64, t_end: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.range(EventKey::time_start(t_start), EventKey::time_end(t_end));
        ewma_to_dict(py, &m)
    }

    fn prefix<'py>(&self, py: Python<'py>, t_end: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.prefix(EventKey::time_end(t_end));
        ewma_to_dict(py, &m)
    }

    fn fork(&self) -> Self {
        Self {
            tree: self.tree.fork(),
        }
    }

    fn merge(&mut self, other: &EwmaStream) {
        self.tree = self.tree.merge(&other.tree);
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }
}

// ---------------------------------------------------------------------------
// 3. CusumStream
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct CusumStream {
    tree: FoldTree<CusumMonoid, 4>,
    mu0: f64,
    sigma0: f64,
    slack_k: f64,
}

#[pymethods]
impl CusumStream {
    #[new]
    #[pyo3(signature = (mu0 = 0.0, sigma0 = 1.0, slack_k = 0.5))]
    fn new(mu0: f64, sigma0: f64, slack_k: f64) -> PyResult<Self> {
        if !mu0.is_finite() || !sigma0.is_finite() || !slack_k.is_finite() {
            return Err(PyValueError::new_err("parameters must be finite"));
        }
        if sigma0 <= 0.0 {
            return Err(PyValueError::new_err("sigma0 must be > 0"));
        }
        if slack_k < 0.0 {
            return Err(PyValueError::new_err("slack_k must be >= 0"));
        }
        Ok(Self {
            tree: FoldTree::new(()),
            mu0,
            sigma0,
            slack_k,
        })
    }

    #[pyo3(signature = (ts, xs, source_id = 0))]
    fn ingest(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        xs: &Bound<'_, PyAny>,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let x_vec = extract_f64_vec(py, xs)?;
        if t_vec.len() != x_vec.len() {
            return Err(PyValueError::new_err("ts and xs must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for (&t, &x) in t_vec.iter().zip(x_vec.iter()) {
                let m = CusumMonoid::singleton(t, x, self.mu0, self.sigma0, self.slack_k)
                    .map_err(to_py_err)?;
                entries.push((EventKey::sourced(t, source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                self.tree = FoldTree::from_sorted((), entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, x, source_id = 0, seq = 0))]
    fn upsert(&mut self, ts: i64, x: f64, source_id: u64, seq: u64) -> PyResult<()> {
        let m = CusumMonoid::singleton(ts, x, self.mu0, self.sigma0, self.slack_k)
            .map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        self.tree.retract(EventKey::sourced(ts, source_id, seq)).is_some()
    }

    fn max_peak(&self) -> f64 {
        self.tree.summary().max_peak()
    }

    fn peak_pos(&self) -> f64 {
        self.tree.summary().peak_pos()
    }

    fn peak_neg(&self) -> f64 {
        self.tree.summary().peak_neg()
    }

    fn current_pos(&self) -> f64 {
        self.tree.summary().current_pos()
    }

    fn current_neg(&self) -> f64 {
        self.tree.summary().current_neg()
    }

    fn onset_ts(&self) -> Option<i64> {
        let t = self.tree.summary().onset_ts();
        if t == i64::MIN { None } else { Some(t) }
    }

    fn peak_ts(&self) -> Option<i64> {
        let t = self.tree.summary().peak_ts();
        if t == i64::MIN { None } else { Some(t) }
    }

    fn first_alarm(&self, threshold: f64) -> Option<i64> {
        self.tree
            .find_first_prefix(|m| m.max_peak() >= threshold)
            .map(|(k, _)| k.event_time)
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        cusum_to_dict(py, &self.tree.summary())
    }

    fn range<'py>(&self, py: Python<'py>, t_start: i64, t_end: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.range(EventKey::time_start(t_start), EventKey::time_end(t_end));
        cusum_to_dict(py, &m)
    }

    fn fork(&self) -> Self {
        Self {
            tree: self.tree.fork(),
            mu0: self.mu0,
            sigma0: self.sigma0,
            slack_k: self.slack_k,
        }
    }

    fn merge(&mut self, other: &CusumStream) {
        self.tree = self.tree.merge(&other.tree);
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }
}

// ---------------------------------------------------------------------------
// 4. EProcessStream
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct EProcessStream {
    tree: FoldTree<EProcessMonoid, 4>,
}

#[pymethods]
impl EProcessStream {
    #[new]
    fn new() -> Self {
        Self {
            tree: FoldTree::new(()),
        }
    }

    #[pyo3(signature = (ts, zs, null_margin = 0.0, source_id = 0))]
    fn ingest_z(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        zs: &Bound<'_, PyAny>,
        null_margin: f64,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let z_vec = extract_f64_vec(py, zs)?;
        if t_vec.len() != z_vec.len() {
            return Err(PyValueError::new_err("ts and zs must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for (&t, &z) in t_vec.iter().zip(z_vec.iter()) {
                let m =
                    EProcessMonoid::singleton_z_composite(t, z, null_margin).map_err(to_py_err)?;
                entries.push((EventKey::sourced(t, source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                self.tree = FoldTree::from_sorted((), entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, xs, mu0, source_id = 0))]
    fn ingest_bounded(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        xs: &Bound<'_, PyAny>,
        mu0: f64,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let x_vec = extract_f64_vec(py, xs)?;
        if t_vec.len() != x_vec.len() {
            return Err(PyValueError::new_err("ts and xs must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for (&t, &x) in t_vec.iter().zip(x_vec.iter()) {
                let m = EProcessMonoid::singleton_bounded(t, x, mu0).map_err(to_py_err)?;
                entries.push((EventKey::sourced(t, source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                self.tree = FoldTree::from_sorted((), entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, z, null_margin = 0.0, source_id = 0, seq = 0))]
    fn upsert_z(&mut self, ts: i64, z: f64, null_margin: f64, source_id: u64, seq: u64) -> PyResult<()> {
        let m = EProcessMonoid::singleton_z_composite(ts, z, null_margin).map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, x, mu0, source_id = 0, seq = 0))]
    fn upsert_bounded(&mut self, ts: i64, x: f64, mu0: f64, source_id: u64, seq: u64) -> PyResult<()> {
        let m = EProcessMonoid::singleton_bounded(ts, x, mu0).map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        self.tree.retract(EventKey::sourced(ts, source_id, seq)).is_some()
    }

    fn log_wealth(&self) -> f64 {
        self.tree.summary().log_wealth()
    }

    fn wealth(&self) -> f64 {
        self.tree.summary().wealth()
    }

    fn log_e_detector(&self) -> f64 {
        self.tree.summary().log_e_detector()
    }

    fn e_detector(&self) -> f64 {
        self.tree.summary().e_detector()
    }

    fn sr_evalue(&self) -> f64 {
        self.tree.summary().sr_evalue()
    }

    fn scan_evalue(&self) -> f64 {
        self.tree.summary().scan_evalue()
    }

    fn e_value(&self) -> f64 {
        self.tree.summary().e_value()
    }

    fn peak_e_detector_valid(&self) -> f64 {
        self.tree.summary().peak_e_detector_valid()
    }

    fn anytime_p_value(&self) -> f64 {
        self.tree.summary().anytime_p_value()
    }

    #[pyo3(signature = (alpha = 0.05))]
    fn rejected(&self, alpha: f64) -> bool {
        self.tree.summary().rejected(alpha)
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        eprocess_to_dict(py, &self.tree.summary())
    }

    fn range<'py>(&self, py: Python<'py>, t_start: i64, t_end: i64) -> PyResult<Bound<'py, PyDict>> {
        let m = self.tree.range(EventKey::time_start(t_start), EventKey::time_end(t_end));
        eprocess_to_dict(py, &m)
    }

    fn fork(&self) -> Self {
        Self {
            tree: self.tree.fork(),
        }
    }

    fn merge(&mut self, other: &EProcessStream) {
        self.tree = self.tree.merge(&other.tree);
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }
}

// ---------------------------------------------------------------------------
// 5. KalmanStream (zero-alloc specialized 1D & 2D + general-dim fallback)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum KalmanInner {
    D1(FoldTree<KalmanFilter1D, 4>),
    D2(FoldTree<KalmanFilter2D, 4>),
    General(FoldTree<KalmanMonoid, 4>),
}

#[pyclass]
#[derive(Clone)]
pub struct KalmanStream {
    dim: usize,
    inner: KalmanInner,
}

#[pymethods]
impl KalmanStream {
    #[new]
    #[pyo3(signature = (dim = 2))]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be >= 1"));
        }
        let inner = match dim {
            1 => KalmanInner::D1(FoldTree::new(())),
            2 => KalmanInner::D2(FoldTree::new(())),
            d => KalmanInner::General(FoldTree::new(d)),
        };
        Ok(Self { dim, inner })
    }

    #[pyo3(signature = (ts, ys, dts, q_accel, r_obs, source_id = 0))]
    fn ingest_constant_velocity(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        ys: &Bound<'_, PyAny>,
        dts: &Bound<'_, PyAny>,
        q_accel: f64,
        r_obs: f64,
        source_id: u64,
    ) -> PyResult<()> {
        let KalmanInner::D2(tree) = &mut self.inner else {
            return Err(PyValueError::new_err("constant_velocity requires dim == 2"));
        };
        let t_vec = extract_i64_vec(py, ts)?;
        let y_vec = extract_f64_vec(py, ys)?;
        let dt_vec = extract_f64_vec(py, dts)?;
        if t_vec.len() != y_vec.len() || t_vec.len() != dt_vec.len() {
            return Err(PyValueError::new_err("ts, ys, and dts must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for i in 0..t_vec.len() {
                let m = KalmanFilter2D::from_constant_velocity(dt_vec[i], q_accel, r_obs, y_vec[i])
                    .map_err(to_py_err)?;
                entries.push((EventKey::sourced(t_vec[i], source_id, 0), m));
            }
            if tree.is_empty() && sorted {
                *tree = FoldTree::from_sorted((), entries);
            } else {
                tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, ys, dts, q_rate, r_obs, source_id = 0))]
    fn ingest_random_walk(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        ys: &Bound<'_, PyAny>,
        dts: &Bound<'_, PyAny>,
        q_rate: f64,
        r_obs: f64,
        source_id: u64,
    ) -> PyResult<()> {
        let KalmanInner::D1(tree) = &mut self.inner else {
            return Err(PyValueError::new_err("random_walk requires dim == 1"));
        };
        let t_vec = extract_i64_vec(py, ts)?;
        let y_vec = extract_f64_vec(py, ys)?;
        let dt_vec = extract_f64_vec(py, dts)?;
        if t_vec.len() != y_vec.len() || t_vec.len() != dt_vec.len() {
            return Err(PyValueError::new_err("ts, ys, and dts must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(t_vec.len());
            for i in 0..t_vec.len() {
                let m = KalmanFilter1D::from_random_walk(dt_vec[i], q_rate, r_obs, y_vec[i])
                    .map_err(to_py_err)?;
                entries.push((EventKey::sourced(t_vec[i], source_id, 0), m));
            }
            if tree.is_empty() && sorted {
                *tree = FoldTree::from_sorted((), entries);
            } else {
                tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, y, dt, q_accel, r_obs, source_id = 0, seq = 0))]
    fn upsert_constant_velocity(
        &mut self,
        ts: i64,
        y: f64,
        dt: f64,
        q_accel: f64,
        r_obs: f64,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        let KalmanInner::D2(tree) = &mut self.inner else {
            return Err(PyValueError::new_err("constant_velocity requires dim == 2"));
        };
        let m = KalmanFilter2D::from_constant_velocity(dt, q_accel, r_obs, y).map_err(to_py_err)?;
        tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, y, dt, q_level, q_slope, r_obs, source_id = 0, seq = 0))]
    fn upsert_local_linear_trend(
        &mut self,
        ts: i64,
        y: f64,
        dt: f64,
        q_level: f64,
        q_slope: f64,
        r_obs: f64,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        let KalmanInner::D2(tree) = &mut self.inner else {
            return Err(PyValueError::new_err("local_linear_trend requires dim == 2"));
        };
        let m = KalmanFilter2D::from_local_linear_trend(dt, q_level, q_slope, r_obs, y)
            .map_err(to_py_err)?;
        tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, f, q, h, r, y, source_id = 0, seq = 0))]
    fn upsert_scalar(
        &mut self,
        ts: i64,
        f: Vec<f64>,
        q: Vec<f64>,
        h: Vec<f64>,
        r: f64,
        y: f64,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        let key = EventKey::sourced(ts, source_id, seq);
        match &mut self.inner {
            KalmanInner::D1(tree) => {
                if f.len() != 1 || q.len() != 1 || h.len() != 1 {
                    return Err(PyValueError::new_err("expected length 1 for dim == 1"));
                }
                let m = KalmanFilter1D::from_observation(f[0], q[0], h[0], r, y).map_err(to_py_err)?;
                tree.upsert(key, m);
            }
            KalmanInner::D2(tree) => {
                if f.len() != 4 || q.len() != 4 || h.len() != 2 {
                    return Err(PyValueError::new_err("expected len(f)=len(q)=4, len(h)=2 for dim == 2"));
                }
                let m = KalmanFilter2D::from_scalar_obs(
                    [f[0], f[1], f[2], f[3]],
                    [q[0], q[1], q[2], q[3]],
                    [h[0], h[1]],
                    r,
                    y,
                )
                .map_err(to_py_err)?;
                tree.upsert(key, m);
            }
            KalmanInner::General(tree) => {
                let m = KalmanMonoid::from_scalar_obs(self.dim, &f, &q, &h, r, y).map_err(to_py_err)?;
                tree.upsert(key, m);
            }
        }
        Ok(())
    }

    #[pyo3(signature = (ts, obs_dim, f, q, h, r, y, source_id = 0, seq = 0))]
    fn upsert_vector(
        &mut self,
        ts: i64,
        obs_dim: usize,
        f: Vec<f64>,
        q: Vec<f64>,
        h: Vec<f64>,
        r: Vec<f64>,
        y: Vec<f64>,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        let key = EventKey::sourced(ts, source_id, seq);
        match &mut self.inner {
            KalmanInner::General(tree) => {
                let m = KalmanMonoid::from_vector_obs(self.dim, obs_dim, &f, &q, &h, &r, &y)
                    .map_err(to_py_err)?;
                tree.upsert(key, m);
                Ok(())
            }
            KalmanInner::D2(tree) if obs_dim == 1 => {
                if f.len() != 4 || q.len() != 4 || h.len() != 2 || r.len() != 1 || y.len() != 1 {
                    return Err(PyValueError::new_err("dimension mismatch for dim == 2, obs_dim == 1"));
                }
                let m = KalmanFilter2D::from_scalar_obs(
                    [f[0], f[1], f[2], f[3]],
                    [q[0], q[1], q[2], q[3]],
                    [h[0], h[1]],
                    r[0],
                    y[0],
                )
                .map_err(to_py_err)?;
                tree.upsert(key, m);
                Ok(())
            }
            _ => {
                // Upgrade D1/D2 to General if vector obs_dim > 1 is requested on dim <= 2
                let m = KalmanMonoid::from_vector_obs(self.dim, obs_dim, &f, &q, &h, &r, &y)
                    .map_err(to_py_err)?;
                let mut gen = FoldTree::new(self.dim);
                match &self.inner {
                    KalmanInner::D1(t) => {
                        for (k, v) in t.iter_all() {
                            gen.upsert(
                                k,
                                KalmanMonoid {
                                    dim: 1,
                                    n: v.n,
                                    a: vec![v.a],
                                    b: vec![v.b],
                                    c: vec![v.c],
                                    eta: vec![v.eta],
                                    j: vec![v.j],
                                    anchor: vec![v.anchor],
                                    log_z: v.log_z,
                                },
                            );
                        }
                    }
                    KalmanInner::D2(t) => {
                        for (k, v) in t.iter_all() {
                            gen.upsert(
                                k,
                                KalmanMonoid {
                                    dim: 2,
                                    n: v.n,
                                    a: v.a.to_vec(),
                                    b: v.b.to_vec(),
                                    c: v.c.to_vec(),
                                    eta: v.eta.to_vec(),
                                    j: v.j.to_vec(),
                                    anchor: v.anchor.to_vec(),
                                    log_z: v.log_z,
                                },
                            );
                        }
                    }
                    KalmanInner::General(_) => unreachable!(),
                }
                gen.upsert(key, m);
                self.inner = KalmanInner::General(gen);
                Ok(())
            }
        }
    }

    #[pyo3(signature = (ts, f, q, source_id = 0, seq = 0))]
    fn upsert_missing(
        &mut self,
        ts: i64,
        f: Vec<f64>,
        q: Vec<f64>,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        let key = EventKey::sourced(ts, source_id, seq);
        match &mut self.inner {
            KalmanInner::D1(tree) => {
                if f.len() != 1 || q.len() != 1 {
                    return Err(PyValueError::new_err("expected length 1 for dim == 1"));
                }
                let m = KalmanFilter1D::from_missing(f[0], q[0]).map_err(to_py_err)?;
                tree.upsert(key, m);
            }
            KalmanInner::D2(tree) => {
                if f.len() != 4 || q.len() != 4 {
                    return Err(PyValueError::new_err("expected len(f)=len(q)=4 for dim == 2"));
                }
                let m = KalmanFilter2D::from_missing(
                    [f[0], f[1], f[2], f[3]],
                    [q[0], q[1], q[2], q[3]],
                )
                .map_err(to_py_err)?;
                tree.upsert(key, m);
            }
            KalmanInner::General(tree) => {
                let m = KalmanMonoid::from_missing(self.dim, &f, &q).map_err(to_py_err)?;
                tree.upsert(key, m);
            }
        }
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        let key = EventKey::sourced(ts, source_id, seq);
        match &mut self.inner {
            KalmanInner::D1(t) => t.retract(key).is_some(),
            KalmanInner::D2(t) => t.retract(key).is_some(),
            KalmanInner::General(t) => t.retract(key).is_some(),
        }
    }

    fn posterior(&self, m0: Vec<f64>, p0: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        match &self.inner {
            KalmanInner::D1(t) => {
                if m0.len() != 1 || p0.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                }
                let (m, p) = t.summary().posterior(m0[0], p0[0]).map_err(to_py_err)?;
                Ok((vec![m], vec![p]))
            }
            KalmanInner::D2(t) => {
                if m0.len() != 2 || p0.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                }
                let (m, p) = t
                    .summary()
                    .posterior([m0[0], m0[1]], [p0[0], p0[1], p0[2], p0[3]])
                    .map_err(to_py_err)?;
                Ok((m.to_vec(), p.to_vec()))
            }
            KalmanInner::General(t) => {
                let post = t.summary().posterior(&m0, &p0).map_err(to_py_err)?;
                Ok((post.mean, post.cov))
            }
        }
    }

    fn range_posterior(
        &self,
        t_start: i64,
        t_end: i64,
        m0: Vec<f64>,
        p0: Vec<f64>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let s = EventKey::time_start(t_start);
        let e = EventKey::time_end(t_end);
        match &self.inner {
            KalmanInner::D1(t) => {
                if m0.len() != 1 || p0.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                }
                let (m, p) = t.range(s, e).posterior(m0[0], p0[0]).map_err(to_py_err)?;
                Ok((vec![m], vec![p]))
            }
            KalmanInner::D2(t) => {
                if m0.len() != 2 || p0.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                }
                let (m, p) = t
                    .range(s, e)
                    .posterior([m0[0], m0[1]], [p0[0], p0[1], p0[2], p0[3]])
                    .map_err(to_py_err)?;
                Ok((m.to_vec(), p.to_vec()))
            }
            KalmanInner::General(t) => {
                let post = t.range(s, e).posterior(&m0, &p0).map_err(to_py_err)?;
                Ok((post.mean, post.cov))
            }
        }
    }

    /// Exact O(log T) Rauch-Tung-Striebel (RTS) smoothed posterior `p(x_t | y_{1:T})` at timestamp `t`.
    fn smooth_at(&self, t: i64, m0: Vec<f64>, p0: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        match &self.inner {
            KalmanInner::D1(tree) => {
                if m0.len() != 1 || p0.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                }
                let (m, p) = smooth_1d(tree, t, m0[0], p0[0]).map_err(to_py_err)?;
                Ok((vec![m], vec![p]))
            }
            KalmanInner::D2(tree) => {
                if m0.len() != 2 || p0.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                }
                let (m, p) =
                    smooth_2d(tree, t, [m0[0], m0[1]], [p0[0], p0[1], p0[2], p0[3]])
                        .map_err(to_py_err)?;
                Ok((m.to_vec(), p.to_vec()))
            }
            KalmanInner::General(tree) => {
                let post = smooth_nd(tree, t, &m0, &p0).map_err(to_py_err)?;
                Ok((post.mean, post.cov))
            }
        }
    }

    /// Batch O(K log T) exact RTS smoothing at multiple query timestamps `ts` in a single Rust loop (GIL-free).
    fn smooth_many(
        &self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        m0: Vec<f64>,
        p0: Vec<f64>,
    ) -> PyResult<SmoothBatch> {
        let t_vec = extract_i64_vec(py, ts)?;
        py.allow_threads(|| {
            let mut means = Vec::with_capacity(t_vec.len());
            let mut covs = Vec::with_capacity(t_vec.len());
            match &self.inner {
                KalmanInner::D1(tree) => {
                    if m0.len() != 1 || p0.len() != 1 {
                        return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                    }
                    for t in t_vec {
                        let (m, p) = smooth_1d(tree, t, m0[0], p0[0]).map_err(to_py_err)?;
                        means.push(vec![m]);
                        covs.push(vec![p]);
                    }
                }
                KalmanInner::D2(tree) => {
                    if m0.len() != 2 || p0.len() != 4 {
                        return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                    }
                    let m0a = [m0[0], m0[1]];
                    let p0a = [p0[0], p0[1], p0[2], p0[3]];
                    for t in t_vec {
                        let (m, p) = smooth_2d(tree, t, m0a, p0a).map_err(to_py_err)?;
                        means.push(m.to_vec());
                        covs.push(p.to_vec());
                    }
                }
                KalmanInner::General(tree) => {
                    for t in t_vec {
                        let post = smooth_nd(tree, t, &m0, &p0).map_err(to_py_err)?;
                        means.push(post.mean);
                        covs.push(post.cov);
                    }
                }
            }
            Ok((means, covs))
        })
    }

    /// Exact Bayesian marginal log-likelihood `log p(y_{1:T})` in O(1) at the root.
    fn log_marginal_likelihood(&self, m0: Vec<f64>, p0: Vec<f64>) -> PyResult<f64> {
        match &self.inner {
            KalmanInner::D1(t) => {
                if m0.len() != 1 || p0.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                }
                t.summary()
                    .log_marginal_likelihood(m0[0], p0[0])
                    .map_err(to_py_err)
            }
            KalmanInner::D2(t) => {
                if m0.len() != 2 || p0.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                }
                t.summary()
                    .log_marginal_likelihood([m0[0], m0[1]], [p0[0], p0[1], p0[2], p0[3]])
                    .map_err(to_py_err)
            }
            KalmanInner::General(t) => t
                .summary()
                .log_marginal_likelihood(&m0, &p0)
                .map_err(to_py_err),
        }
    }

    /// Exact Bayesian marginal log-likelihood `log p(y_{t_start:t_end})` in O(log T).
    fn range_log_marginal_likelihood(
        &self,
        t_start: i64,
        t_end: i64,
        m0: Vec<f64>,
        p0: Vec<f64>,
    ) -> PyResult<f64> {
        let s = EventKey::time_start(t_start);
        let e = EventKey::time_end(t_end);
        match &self.inner {
            KalmanInner::D1(t) => {
                if m0.len() != 1 || p0.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                }
                t.range(s, e)
                    .log_marginal_likelihood(m0[0], p0[0])
                    .map_err(to_py_err)
            }
            KalmanInner::D2(t) => {
                if m0.len() != 2 || p0.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                }
                t.range(s, e)
                    .log_marginal_likelihood([m0[0], m0[1]], [p0[0], p0[1], p0[2], p0[3]])
                    .map_err(to_py_err)
            }
            KalmanInner::General(t) => t
                .range(s, e)
                .log_marginal_likelihood(&m0, &p0)
                .map_err(to_py_err),
        }
    }

    /// Exact O(log T) Bayesian changepoint log-Bayes-factor at split timestamp `t`.
    ///
    /// If `q_jump` is `None`, tests a full state reset to `N(m0, P0)` (`log(p(y_{<=t})p(y_{>t})/p(y_{1:T}))`).
    /// If `q_jump` is provided (length `dim * dim`), tests a translation-invariant instantaneous
    /// state shock `N(m_f, P_f + Q_jump)` vs continuity `N(m_f, P_f)`.
    #[pyo3(signature = (t, m0, p0, q_jump = None))]
    fn changepoint_bayes_factor(
        &self,
        t: i64,
        m0: Vec<f64>,
        p0: Vec<f64>,
        q_jump: Option<Vec<f64>>,
    ) -> PyResult<f64> {
        match (&self.inner, q_jump) {
            (KalmanInner::D1(tree), None) => {
                if m0.len() != 1 || p0.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                }
                changepoint_bayes_factor_1d(tree, t, m0[0], p0[0]).map_err(to_py_err)
            }
            (KalmanInner::D1(tree), Some(qj)) => {
                if m0.len() != 1 || p0.len() != 1 || qj.len() != 1 {
                    return Err(PyValueError::new_err("expected len(m0)=len(p0)=len(q_jump)=1"));
                }
                changepoint_shock_bayes_factor_1d(tree, t, m0[0], p0[0], qj[0]).map_err(to_py_err)
            }
            (KalmanInner::D2(tree), None) => {
                if m0.len() != 2 || p0.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                }
                changepoint_bayes_factor_2d(
                    tree,
                    t,
                    [m0[0], m0[1]],
                    [p0[0], p0[1], p0[2], p0[3]],
                )
                .map_err(to_py_err)
            }
            (KalmanInner::D2(tree), Some(qj)) => {
                if m0.len() != 2 || p0.len() != 4 || qj.len() != 4 {
                    return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=len(q_jump)=4"));
                }
                changepoint_shock_bayes_factor_2d(
                    tree,
                    t,
                    [m0[0], m0[1]],
                    [p0[0], p0[1], p0[2], p0[3]],
                    [qj[0], qj[1], qj[2], qj[3]],
                )
                .map_err(to_py_err)
            }
            (KalmanInner::General(tree), None) => {
                changepoint_bayes_factor_nd(tree, t, &m0, &p0).map_err(to_py_err)
            }
            (KalmanInner::General(tree), Some(qj)) => {
                changepoint_shock_bayes_factor_nd(tree, t, &m0, &p0, &qj).map_err(to_py_err)
            }
        }
    }

    /// Batch O(K log T) GIL-free Bayesian changepoint scan across candidate split timestamps `ts`.
    #[pyo3(signature = (ts, m0, p0, q_jump = None))]
    fn scan_changepoints<'py>(
        &self,
        py: Python<'py>,
        ts: &Bound<'_, PyAny>,
        m0: Vec<f64>,
        p0: Vec<f64>,
        q_jump: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let t_vec = extract_i64_vec(py, ts)?;
        let (best_ts, best_bf, bfs) = py.allow_threads(|| -> PyResult<(Option<i64>, f64, Vec<f64>)> {
            let mut bfs = Vec::with_capacity(t_vec.len());
            let mut best_ts = None;
            let mut best_bf = f64::NEG_INFINITY;
            match (&self.inner, q_jump.as_deref()) {
                (KalmanInner::D1(tree), None) => {
                    if m0.len() != 1 || p0.len() != 1 {
                        return Err(PyValueError::new_err("expected len(m0)=1, len(p0)=1"));
                    }
                    for &t in &t_vec {
                        let bf =
                            changepoint_bayes_factor_1d(tree, t, m0[0], p0[0]).map_err(to_py_err)?;
                        if bf > best_bf {
                            best_bf = bf;
                            best_ts = Some(t);
                        }
                        bfs.push(bf);
                    }
                }
                (KalmanInner::D1(tree), Some(qj)) => {
                    if m0.len() != 1 || p0.len() != 1 || qj.len() != 1 {
                        return Err(PyValueError::new_err("expected len(m0)=len(p0)=len(q_jump)=1"));
                    }
                    for &t in &t_vec {
                        let bf = changepoint_shock_bayes_factor_1d(tree, t, m0[0], p0[0], qj[0])
                            .map_err(to_py_err)?;
                        if bf > best_bf {
                            best_bf = bf;
                            best_ts = Some(t);
                        }
                        bfs.push(bf);
                    }
                }
                (KalmanInner::D2(tree), None) => {
                    if m0.len() != 2 || p0.len() != 4 {
                        return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=4"));
                    }
                    let m0a = [m0[0], m0[1]];
                    let p0a = [p0[0], p0[1], p0[2], p0[3]];
                    for &t in &t_vec {
                        let bf =
                            changepoint_bayes_factor_2d(tree, t, m0a, p0a).map_err(to_py_err)?;
                        if bf > best_bf {
                            best_bf = bf;
                            best_ts = Some(t);
                        }
                        bfs.push(bf);
                    }
                }
                (KalmanInner::D2(tree), Some(qj)) => {
                    if m0.len() != 2 || p0.len() != 4 || qj.len() != 4 {
                        return Err(PyValueError::new_err("expected len(m0)=2, len(p0)=len(q_jump)=4"));
                    }
                    let m0a = [m0[0], m0[1]];
                    let p0a = [p0[0], p0[1], p0[2], p0[3]];
                    let qja = [qj[0], qj[1], qj[2], qj[3]];
                    for &t in &t_vec {
                        let bf = changepoint_shock_bayes_factor_2d(tree, t, m0a, p0a, qja)
                            .map_err(to_py_err)?;
                        if bf > best_bf {
                            best_bf = bf;
                            best_ts = Some(t);
                        }
                        bfs.push(bf);
                    }
                }
                (KalmanInner::General(tree), None) => {
                    for &t in &t_vec {
                        let bf =
                            changepoint_bayes_factor_nd(tree, t, &m0, &p0).map_err(to_py_err)?;
                        if bf > best_bf {
                            best_bf = bf;
                            best_ts = Some(t);
                        }
                        bfs.push(bf);
                    }
                }
                (KalmanInner::General(tree), Some(qj)) => {
                    for &t in &t_vec {
                        let bf = changepoint_shock_bayes_factor_nd(tree, t, &m0, &p0, qj)
                            .map_err(to_py_err)?;
                        if bf > best_bf {
                            best_bf = bf;
                            best_ts = Some(t);
                        }
                        bfs.push(bf);
                    }
                }
            }
            Ok((best_ts, if best_ts.is_some() { best_bf } else { 0.0 }, bfs))
        })?;
        let d = PyDict::new(py);
        d.set_item("best_ts", best_ts)?;
        d.set_item("best_log_bf", best_bf)?;
        d.set_item("log_bfs", bfs)?;
        Ok(d)
    }

    /// Zero-blind-spot O(log^2 T) GIL-free global Bayesian shock changepoint locator.
    ///
    /// Screens overlapping FoldTree segments by excess innovation NLL, zooms into top-3 regions,
    /// and pinpoints the exact shock timestamp & log-Bayes-factor in sub-millisecond time.
    #[pyo3(signature = (m0, p0, q_jump, n_windows = 64))]
    fn locate_changepoint<'py>(
        &self,
        py: Python<'py>,
        m0: Vec<f64>,
        p0: Vec<f64>,
        q_jump: Vec<f64>,
        n_windows: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let hit = py.allow_threads(|| -> PyResult<Option<(i64, f64)>> {
            match &self.inner {
                KalmanInner::D1(tree) => {
                    if m0.len() != 1 || p0.len() != 1 || q_jump.len() != 1 {
                        return Err(PyValueError::new_err(
                            "expected len(m0)=len(p0)=len(q_jump)=1",
                        ));
                    }
                    locate_changepoint_1d(tree, m0[0], p0[0], q_jump[0], n_windows)
                        .map_err(to_py_err)
                }
                KalmanInner::D2(tree) => {
                    if m0.len() != 2 || p0.len() != 4 || q_jump.len() != 4 {
                        return Err(PyValueError::new_err(
                            "expected len(m0)=2, len(p0)=len(q_jump)=4",
                        ));
                    }
                    locate_changepoint_2d(
                        tree,
                        [m0[0], m0[1]],
                        [p0[0], p0[1], p0[2], p0[3]],
                        [q_jump[0], q_jump[1], q_jump[2], q_jump[3]],
                        n_windows,
                    )
                    .map_err(to_py_err)
                }
                KalmanInner::General(_) => Err(PyValueError::new_err(
                    "locate_changepoint is currently specialized for dim 1 and dim 2; use scan_changepoints for general dim",
                )),
            }
        })?;
        let d = PyDict::new(py);
        match hit {
            Some((t, bf)) => {
                d.set_item("best_ts", Some(t))?;
                d.set_item("best_log_bf", bf)?;
            }
            None => {
                d.set_item("best_ts", None::<i64>)?;
                d.set_item("best_log_bf", 0.0)?;
            }
        }
        Ok(d)
    }

    fn fork(&self) -> Self {
        let inner = match &self.inner {
            KalmanInner::D1(t) => KalmanInner::D1(t.fork()),
            KalmanInner::D2(t) => KalmanInner::D2(t.fork()),
            KalmanInner::General(t) => KalmanInner::General(t.fork()),
        };
        Self {
            dim: self.dim,
            inner,
        }
    }

    fn merge(&mut self, other: &KalmanStream) -> PyResult<()> {
        if self.dim != other.dim {
            return Err(PyValueError::new_err("cannot merge KalmanStreams of different dimensions"));
        }
        match (&mut self.inner, &other.inner) {
            (KalmanInner::D1(a), KalmanInner::D1(b)) => *a = a.merge(b),
            (KalmanInner::D2(a), KalmanInner::D2(b)) => *a = a.merge(b),
            (KalmanInner::General(a), KalmanInner::General(b)) => *a = a.merge(b),
            _ => {
                return Err(PyValueError::new_err(
                    "cannot merge specialized and vector-upgraded KalmanStreams",
                ))
            }
        }
        Ok(())
    }

    fn __len__(&self) -> usize {
        match &self.inner {
            KalmanInner::D1(t) => t.len(),
            KalmanInner::D2(t) => t.len(),
            KalmanInner::General(t) => t.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// 6. RidgeStream
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct RidgeStream {
    dim: usize,
    tree: FoldTree<RidgeMonoid, 4>,
}

#[pymethods]
impl RidgeStream {
    #[new]
    #[pyo3(signature = (dim, decay_rate = 0.0, half_life = None))]
    fn new(dim: usize, decay_rate: f64, half_life: Option<f64>) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be >= 1"));
        }
        let lambda = if let Some(hl) = half_life {
            EwmaMonoid::decay_rate_from_half_life(hl).map_err(to_py_err)?
        } else {
            if !decay_rate.is_finite() || decay_rate < 0.0 {
                return Err(PyValueError::new_err("decay_rate must be finite and >= 0"));
            }
            decay_rate
        };
        Ok(Self {
            dim,
            tree: FoldTree::new((dim, lambda)),
        })
    }

    #[pyo3(signature = (ts, xmat, yvec, source_id = 0))]
    fn ingest(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        xmat: &Bound<'_, PyAny>,
        yvec: &Bound<'_, PyAny>,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let (x_flat, n_rows) = extract_f64_mat(py, xmat, self.dim)?;
        let y_vec = extract_f64_vec(py, yvec)?;
        if t_vec.len() != n_rows || t_vec.len() != y_vec.len() {
            return Err(PyValueError::new_err("ts, xmat rows, and yvec must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let row = &x_flat[i * self.dim..(i + 1) * self.dim];
                let m =
                    RidgeMonoid::singleton(t_vec[i], row, y_vec[i], 1.0).map_err(to_py_err)?;
                entries.push((EventKey::sourced(t_vec[i], source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                let ctx = *self.tree.ctx();
                self.tree = FoldTree::from_sorted(ctx, entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, x, y, weight = 1.0, source_id = 0, seq = 0))]
    fn upsert(
        &mut self,
        ts: i64,
        x: Vec<f64>,
        y: f64,
        weight: f64,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        if x.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "expected feature len {}, got {}",
                self.dim,
                x.len()
            )));
        }
        let m = RidgeMonoid::singleton(ts, &x, y, weight).map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        self.tree.retract(EventKey::sourced(ts, source_id, seq)).is_some()
    }

    #[pyo3(signature = (alpha = 1e-4))]
    fn solve<'py>(&self, py: Python<'py>, alpha: f64) -> PyResult<Bound<'py, PyDict>> {
        let sol = self.tree.summary().solve(alpha).map_err(to_py_err)?;
        ridge_sol_to_dict(py, &sol)
    }

    #[pyo3(signature = (x, alpha = 1e-4))]
    fn predict_with_variance(&self, x: Vec<f64>, alpha: f64) -> PyResult<(f64, f64)> {
        self.tree
            .summary()
            .predict_with_variance(&x, alpha)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (t_start, t_end, alpha = 1e-4))]
    fn range_solve<'py>(
        &self,
        py: Python<'py>,
        t_start: i64,
        t_end: i64,
        alpha: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sol = self
            .tree
            .range(EventKey::time_start(t_start), EventKey::time_end(t_end))
            .solve(alpha)
            .map_err(to_py_err)?;
        ridge_sol_to_dict(py, &sol)
    }

    fn fork(&self) -> Self {
        Self {
            dim: self.dim,
            tree: self.tree.fork(),
        }
    }

    fn merge(&mut self, other: &RidgeStream) -> PyResult<()> {
        if self.dim != other.dim {
            return Err(PyValueError::new_err("cannot merge RidgeStreams of different dimensions"));
        }
        self.tree = self.tree.merge(&other.tree);
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }
}

// ---------------------------------------------------------------------------
// 6b. CovarianceStream
// ---------------------------------------------------------------------------

fn flat_to_rows(flat: &[f64], dim: usize) -> Vec<Vec<f64>> {
    flat.chunks_exact(dim).map(|r| r.to_vec()).collect()
}

#[pyclass]
#[derive(Clone)]
pub struct CovarianceStream {
    dim: usize,
    tree: FoldTree<CovarianceMonoid, 4>,
}

#[pymethods]
impl CovarianceStream {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be >= 1"));
        }
        Ok(Self {
            dim,
            tree: FoldTree::new(dim),
        })
    }

    #[pyo3(signature = (ts, xmat, source_id = 0))]
    fn ingest(
        &mut self,
        py: Python<'_>,
        ts: &Bound<'_, PyAny>,
        xmat: &Bound<'_, PyAny>,
        source_id: u64,
    ) -> PyResult<()> {
        let t_vec = extract_i64_vec(py, ts)?;
        let (x_flat, n_rows) = extract_f64_mat(py, xmat, self.dim)?;
        if t_vec.len() != n_rows {
            return Err(PyValueError::new_err("ts and xmat rows must have equal length"));
        }
        py.allow_threads(|| {
            let sorted = is_non_decreasing(&t_vec);
            let mut entries = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let row = &x_flat[i * self.dim..(i + 1) * self.dim];
                let m = CovarianceMonoid::singleton(t_vec[i], row).map_err(to_py_err)?;
                entries.push((EventKey::sourced(t_vec[i], source_id, 0), m));
            }
            if self.tree.is_empty() && sorted {
                let ctx = *self.tree.ctx();
                self.tree = FoldTree::from_sorted(ctx, entries);
            } else {
                self.tree.upsert_batch(entries);
            }
            Ok(())
        })
    }

    #[pyo3(signature = (ts, x, source_id = 0, seq = 0))]
    fn upsert(
        &mut self,
        ts: i64,
        x: Vec<f64>,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        if x.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "expected vector len {}, got {}",
                self.dim,
                x.len()
            )));
        }
        let m = CovarianceMonoid::singleton(ts, &x).map_err(to_py_err)?;
        self.tree.upsert(EventKey::sourced(ts, source_id, seq), m);
        Ok(())
    }

    #[pyo3(signature = (ts, source_id = 0, seq = 0))]
    fn retract(&mut self, ts: i64, source_id: u64, seq: u64) -> bool {
        self.tree.retract(EventKey::sourced(ts, source_id, seq)).is_some()
    }

    fn mean(&self) -> Vec<f64> {
        self.tree.summary().mean().to_vec()
    }

    #[pyo3(signature = (ddof = 1))]
    fn covariance(&self, ddof: u64) -> Vec<Vec<f64>> {
        let flat = self.tree.summary().covariance(ddof);
        flat_to_rows(&flat, self.dim)
    }

    fn correlation(&self) -> Vec<Vec<f64>> {
        let flat = self.tree.summary().correlation();
        flat_to_rows(&flat, self.dim)
    }

    /// Exact cyclic-Jacobi PCA of the full compensated streaming covariance matrix (zero BLAS).
    #[pyo3(signature = (ddof = 1, top_k = None))]
    fn pca<'py>(&self, py: Python<'py>, ddof: u64, top_k: Option<usize>) -> PyResult<Bound<'py, PyDict>> {
        let res = self.tree.summary().pca(ddof, top_k);
        pca_to_dict(py, &res)
    }

    /// Exact O(log T + d^3) cyclic-Jacobi PCA over any time range `[t_start, t_end]`.
    #[pyo3(signature = (t_start, t_end, ddof = 1, top_k = None))]
    fn range_pca<'py>(
        &self,
        py: Python<'py>,
        t_start: i64,
        t_end: i64,
        ddof: u64,
        top_k: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let res = self
            .tree
            .range(EventKey::time_start(t_start), EventKey::time_end(t_end))
            .pca(ddof, top_k);
        pca_to_dict(py, &res)
    }

    #[pyo3(signature = (x, ridge_eps = 1e-6))]
    fn mahalanobis_sq(&self, x: Vec<f64>, ridge_eps: f64) -> PyResult<f64> {
        self.tree
            .summary()
            .mahalanobis_sq(&x, ridge_eps)
            .map_err(to_py_err)
    }

    #[pyo3(signature = (xmat, ridge_eps = 1e-6))]
    fn mahalanobis_many(
        &self,
        py: Python<'_>,
        xmat: &Bound<'_, PyAny>,
        ridge_eps: f64,
    ) -> PyResult<Vec<f64>> {
        let (x_flat, _) = extract_f64_mat(py, xmat, self.dim)?;
        py.allow_threads(|| {
            self.tree
                .summary()
                .mahalanobis_sq_batch(&x_flat, ridge_eps)
                .map_err(to_py_err)
        })
    }

    #[pyo3(signature = (t_start, t_end, ddof = 1))]
    fn range_covariance(&self, t_start: i64, t_end: i64, ddof: u64) -> Vec<Vec<f64>> {
        let flat = self
            .tree
            .range(EventKey::time_start(t_start), EventKey::time_end(t_end))
            .covariance(ddof);
        flat_to_rows(&flat, self.dim)
    }

    fn range_correlation(&self, t_start: i64, t_end: i64) -> Vec<Vec<f64>> {
        let flat = self
            .tree
            .range(EventKey::time_start(t_start), EventKey::time_end(t_end))
            .correlation();
        flat_to_rows(&flat, self.dim)
    }

    #[pyo3(signature = (t_start, t_end, x, ridge_eps = 1e-6))]
    fn range_mahalanobis_sq(
        &self,
        t_start: i64,
        t_end: i64,
        x: Vec<f64>,
        ridge_eps: f64,
    ) -> PyResult<f64> {
        self.tree
            .range(EventKey::time_start(t_start), EventKey::time_end(t_end))
            .mahalanobis_sq(&x, ridge_eps)
            .map_err(to_py_err)
    }

    fn fork(&self) -> Self {
        Self {
            dim: self.dim,
            tree: self.tree.fork(),
        }
    }

    fn merge(&mut self, other: &CovarianceStream) -> PyResult<()> {
        if self.dim != other.dim {
            return Err(PyValueError::new_err(
                "cannot merge CovarianceStreams of different dimensions",
            ));
        }
        self.tree = self.tree.merge(&other.tree);
        Ok(())
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }
}

// ---------------------------------------------------------------------------
// 7. SwarmEProcessLedger & BitemporalMoments
// ---------------------------------------------------------------------------

#[pyclass]
#[derive(Clone)]
pub struct SwarmEProcessLedger {
    inner: SwarmLedger<EProcessMonoid, 4>,
}

#[pymethods]
impl SwarmEProcessLedger {
    #[new]
    fn new() -> Self {
        Self {
            inner: SwarmLedger::new(()),
        }
    }

    fn spawn_agent(&mut self, agent_id: u64) {
        self.inner.spawn_agent(agent_id);
    }

    #[pyo3(signature = (agent_id, ts, source_id, seq, z, null_margin = 0.0))]
    fn observe_z(
        &mut self,
        agent_id: u64,
        ts: i64,
        source_id: u64,
        seq: u64,
        z: f64,
        null_margin: f64,
    ) -> PyResult<bool> {
        let m = EProcessMonoid::singleton_z_composite(ts, z, null_margin).map_err(to_py_err)?;
        Ok(self.inner.observe(agent_id, EventKey::sourced(ts, source_id, seq), m))
    }

    fn retract(&mut self, agent_id: u64, ts: i64, source_id: u64, seq: u64) -> bool {
        self.inner.retract(agent_id, EventKey::sourced(ts, source_id, seq))
    }

    fn agent_summary<'py>(&self, py: Python<'py>, agent_id: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.agent_view(agent_id) {
            Some(view) => Ok(Some(eprocess_to_dict(py, &view.summary())?)),
            None => Ok(None),
        }
    }

    fn merge_all<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let summary = self.inner.merge_all().summary();
        eprocess_to_dict(py, &summary)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new(py);
        d.set_item("n_agents", s.n_agents)?;
        d.set_item("raw_observations", s.raw_observations)?;
        d.set_item("unique_events", s.unique_events)?;
        d.set_item("echoes_deduped", s.echoes_deduped)?;
        d.set_item("retractions", s.retractions)?;
        d.set_item("overwrites", s.overwrites)?;
        Ok(d)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct BitemporalMoments {
    inner: BitemporalFold<MomentsMonoid, 4>,
}

#[pymethods]
impl BitemporalMoments {
    #[new]
    fn new() -> Self {
        Self {
            inner: BitemporalFold::new(()),
        }
    }

    #[pyo3(signature = (event_time, knowledge_time, x, source_id = 0, seq = 0))]
    fn ingest_event(
        &mut self,
        event_time: i64,
        knowledge_time: i64,
        x: f64,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        let m = MomentsMonoid::singleton(event_time, x).map_err(to_py_err)?;
        self.inner
            .upsert(EventKey::sourced(event_time, source_id, seq), m);
        self.inner.commit(knowledge_time).map_err(to_py_err)?;
        Ok(())
    }

    #[pyo3(signature = (event_time, knowledge_time, source_id = 0, seq = 0))]
    fn retract_event(
        &mut self,
        event_time: i64,
        knowledge_time: i64,
        source_id: u64,
        seq: u64,
    ) -> PyResult<()> {
        self.inner
            .retract(EventKey::sourced(event_time, source_id, seq));
        self.inner.commit(knowledge_time).map_err(to_py_err)?;
        Ok(())
    }

    fn as_of<'py>(&self, py: Python<'py>, knowledge_time: i64) -> PyResult<Bound<'py, PyDict>> {
        match self.inner.as_of(knowledge_time) {
            Some(snap) => moments_to_dict(py, &snap.summary()),
            None => {
                use backfold_core::Monoid;
                moments_to_dict(py, &MomentsMonoid::identity(&()))
            }
        }
    }

    fn live_summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        moments_to_dict(py, &self.inner.current().summary())
    }
}

// ---------------------------------------------------------------------------
// 8. Top-level functions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (e_values, alpha = 0.05))]
fn ebh_reject(py: Python<'_>, e_values: &Bound<'_, PyAny>, alpha: f64) -> PyResult<Vec<usize>> {
    let es = extract_f64_vec(py, e_values)?;
    Ok(core_ebh_reject(&es, alpha))
}

#[pyfunction]
#[pyo3(signature = (channel, ts, xs, top_k = 5, threshold = 10.0))]
fn digest_stream<'py>(
    py: Python<'py>,
    channel: &str,
    ts: &Bound<'_, PyAny>,
    xs: &Bound<'_, PyAny>,
    top_k: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let t_vec = extract_i64_vec(py, ts)?;
    let x_vec = extract_f64_vec(py, xs)?;
    let ch = channel.to_string();
    let dig = py
        .allow_threads(|| build_stream_digest(&ch, &t_vec, &x_vec, top_k, threshold))
        .map_err(to_py_err)?;
    digest_to_dict(py, &dig)
}

#[pyfunction]
#[pyo3(signature = (channels_2d, warmup = 120, alpha = 0.05, null_margin = 0.15))]
fn sift_streams<'py>(
    py: Python<'py>,
    channels_2d: &Bound<'_, PyAny>,
    warmup: usize,
    alpha: f64,
    null_margin: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (flat, n_streams, t_len) = extract_f64_2d(py, channels_2d)?;
    let res = py
        .allow_threads(|| {
            core_sift_streams_fdr(&flat, n_streams, t_len, warmup, alpha, null_margin)
        })
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("rejected", res.rejected_indices)?;
    d.set_item("e_values", res.e_values)?;
    d.set_item("baselines", res.baselines)?;
    Ok(d)
}

#[pyfunction]
fn causal_robust_scale(py: Python<'_>, prefix: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let v = extract_f64_vec(py, prefix)?;
    Ok(core_robust_scale(&v))
}

#[pymodule]
fn _backfold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<MomentsStream>()?;
    m.add_class::<EwmaStream>()?;
    m.add_class::<CusumStream>()?;
    m.add_class::<EProcessStream>()?;
    m.add_class::<KalmanStream>()?;
    m.add_class::<RidgeStream>()?;
    m.add_class::<CovarianceStream>()?;
    m.add_class::<SwarmEProcessLedger>()?;
    m.add_class::<BitemporalMoments>()?;
    m.add_function(wrap_pyfunction!(ebh_reject, m)?)?;
    m.add_function(wrap_pyfunction!(digest_stream, m)?)?;
    m.add_function(wrap_pyfunction!(sift_streams, m)?)?;
    m.add_function(wrap_pyfunction!(causal_robust_scale, m)?)?;
    Ok(())
}
