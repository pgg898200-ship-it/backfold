//! Zero-dependency WebAssembly interactive engine for `backfold`.
//!
//! Compiles directly to `wasm32-unknown-unknown` with `cargo build --release --target wasm32-unknown-unknown`
//! (zero `wasm-bindgen` / `wasm-pack` version lock-in). Exposes a minimal JSON-over-linear-memory
//! ABI (`bf_alloc`, `bf_dealloc`, `bf_exec`) backed by real `backfold-core` persistent `FoldTree`s:
//! - Exact O(log T) 2D Constant-Velocity Kalman filter + Rauch-Tung-Striebel smoother
//! - Exact O(1) root marginal log-likelihood `log p(y_{1:T})`
//! - Exact O(K log T) translation-invariant Bayesian shock changepoint scan
//! - Exact <300-token lossless-recoverable agent stream digest

use std::cell::RefCell;

use backfold_core::digest::build_stream_digest;
use backfold_core::ops::{changepoint_shock_bayes_factor_2d, smooth_2d, KalmanFilter2D};
use backfold_core::{EventKey, FoldTree};
use serde::{Deserialize, Serialize};

/// Deterministic splitmix64 PRNG matching CLI benchmarks.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

struct Session {
    n: usize,
    q_accel: f64,
    r_obs: f64,
    m0: [f64; 2],
    p0: [f64; 4],
    ts: Vec<i64>,
    ys: Vec<f64>,
    orig_ys: Vec<f64>,
    retracted: Vec<bool>,
    tree: FoldTree<KalmanFilter2D, 4>,
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

#[derive(Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum Command {
    Init {
        n: usize,
        seed: u64,
        shock_t: Option<usize>,
        shock_step: Option<f64>,
        shock_vel: Option<f64>,
        q_accel: Option<f64>,
        r_obs: Option<f64>,
        n_plot: Option<usize>,
        n_scan: Option<usize>,
    },
    Upsert {
        t: usize,
        y: f64,
        query_t: Option<usize>,
        n_plot: Option<usize>,
        n_scan: Option<usize>,
    },
    Retract {
        t: usize,
        query_t: Option<usize>,
        n_plot: Option<usize>,
        n_scan: Option<usize>,
    },
    Restore {
        t: usize,
        query_t: Option<usize>,
        n_plot: Option<usize>,
        n_scan: Option<usize>,
    },
    BenchEdit {
        t: usize,
        y_alt: f64,
        query_t: usize,
        reps: usize,
    },
    Digest {
        top_k: Option<usize>,
        threshold: Option<f64>,
    },
    GetRaw,
}

#[derive(Serialize)]
struct PlotPoint {
    t: usize,
    y: f64,
    retracted: bool,
    filtered: f64,
    smoothed: f64,
    smooth_std: f64,
}

#[derive(Serialize)]
struct ScanPoint {
    t: usize,
    log_bf: f64,
}

#[derive(Serialize)]
struct SnapshotResp {
    ok: bool,
    n: usize,
    active_len: usize,
    query_t: usize,
    query_smooth_mean: [f64; 2],
    query_smooth_cov: [f64; 4],
    filtered_end: [f64; 2],
    log_marginal_likelihood: f64,
    best_changepoint_t: usize,
    best_log_bf: f64,
    plot: Vec<PlotPoint>,
    scan: Vec<ScanPoint>,
}

fn build_snapshot(
    sess: &Session,
    query_t: usize,
    n_plot: usize,
    n_scan: usize,
) -> SnapshotResp {
    let qt = query_t.min(sess.n.saturating_sub(1));
    let (qm, qc) = smooth_2d(&sess.tree, qt as i64, sess.m0, sess.p0).unwrap_or(([0.0; 2], [0.0; 4]));
    let (fm_end, _) = sess.tree.summary().posterior(sess.m0, sess.p0).unwrap_or(([0.0; 2], [0.0; 4]));
    let log_lik = sess
        .tree
        .summary()
        .log_marginal_likelihood(sess.m0, sess.p0)
        .unwrap_or(0.0);

    let plot_cnt = n_plot.clamp(16, 240).min(sess.n);
    let mut plot = Vec::with_capacity(plot_cnt);
    for i in 0..plot_cnt {
        let idx = if plot_cnt <= 1 {
            0
        } else {
            (i * (sess.n - 1)) / (plot_cnt - 1)
        };
        let (left, right) = sess.tree.split_summary(EventKey::time_end(idx as i64));
        let (fm, fp) = left.posterior(sess.m0, sess.p0).unwrap_or((sess.m0, sess.p0));
        let (sm, sp) = KalmanFilter2D::smooth_fusion(fm, fp, &right).unwrap_or((fm, fp));
        plot.push(PlotPoint {
            t: idx,
            y: sess.ys[idx],
            retracted: sess.retracted[idx],
            filtered: fm[0],
            smoothed: sm[0],
            smooth_std: sp[0].max(0.0).sqrt(),
        });
    }

    let scan_cnt = n_scan.clamp(8, 128);
    let q_jump = [40.0 * sess.r_obs.max(0.1), 0.0, 0.0, 0.5];
    let (best_t_i64, best_bf) =
        backfold_core::ops::locate_changepoint_2d(&sess.tree, sess.m0, sess.p0, q_jump, scan_cnt)
            .unwrap_or(None)
            .unwrap_or((sess.n as i64 / 2, 0.0));
    let best_t = best_t_i64.max(0) as usize;

    // Build window-coverage profile so any shock anywhere in [0, n) lights up cleanly!
    let baseline = sess.tree.summary().log_z / (sess.n.max(1) as f64);
    let mut scan = Vec::with_capacity(scan_cnt);
    for k in 0..scan_cnt {
        let a = (k * sess.n) / scan_cnt;
        let b = (((k + 1) * sess.n) / scan_cnt).min(sess.n.saturating_sub(1));
        let mid = (a + b) / 2;
        let mut score = if best_t >= a && best_t <= b {
            best_bf
        } else {
            let seg = sess.tree.query_time_range(a as i64, b as i64);
            let excess = if seg.n >= 2 {
                baseline * (seg.n as f64) - seg.log_z
            } else {
                0.0
            };
            let pt_bf = changepoint_shock_bayes_factor_2d(
                &sess.tree,
                mid as i64,
                sess.m0,
                sess.p0,
                q_jump,
            )
            .unwrap_or(0.0);
            pt_bf.max(excess)
        };
        if !score.is_finite() {
            score = 0.0;
        }
        let rep_t = if best_t >= a && best_t <= b { best_t } else { mid };
        scan.push(ScanPoint {
            t: rep_t,
            log_bf: score,
        });
    }

    SnapshotResp {
        ok: true,
        n: sess.n,
        active_len: sess.tree.len(),
        query_t: qt,
        query_smooth_mean: qm,
        query_smooth_cov: qc,
        filtered_end: fm_end,
        log_marginal_likelihood: log_lik,
        best_changepoint_t: best_t,
        best_log_bf: best_bf,
        plot,
        scan,
    }
}

/// Core JSON command dispatcher (callable natively in Rust tests or via WASM FFI).
pub fn exec_json(input: &str) -> String {
    let cmd: Command = match serde_json::from_str(input) {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({ "ok": false, "error": format!("bad command: {e}") })
                .to_string();
        }
    };

    SESSION.with(|cell| {
        let mut slot = cell.borrow_mut();
        match cmd {
            Command::Init {
                n,
                seed,
                shock_t,
                shock_step,
                shock_vel,
                q_accel,
                r_obs,
                n_plot,
                n_scan,
            } => {
                let n = n.clamp(32, 100_000);
                let q = q_accel.unwrap_or(1e-3).max(1e-9);
                let r = r_obs.unwrap_or(0.25).max(1e-9);
                let st = shock_t.unwrap_or(n / 2);
                let step = shock_step.unwrap_or(6.5);
                let vel = shock_vel.unwrap_or(0.0015);
                let mut rng = Rng::new(seed);
                let mut ts = Vec::with_capacity(n);
                let mut ys = Vec::with_capacity(n);
                let mut entries = Vec::with_capacity(n);
                let mut level = 0.0f64;
                let mut slope = 0.002f64;
                for i in 0..n {
                    if i == st {
                        level += step;
                        slope += vel;
                    }
                    level += slope + rng.normal() * (q.sqrt());
                    let y = level + rng.normal() * r.sqrt();
                    ts.push(i as i64);
                    ys.push(y);
                    let leaf = KalmanFilter2D::from_constant_velocity(1.0, q, r, y).unwrap();
                    entries.push((EventKey::at(i as i64), leaf));
                }
                let orig_ys = ys.clone();
                let tree = FoldTree::from_sorted((), entries);
                let sess = Session {
                    n,
                    q_accel: q,
                    r_obs: r,
                    m0: [0.0, 0.0],
                    p0: [10.0, 0.0, 0.0, 10.0],
                    ts,
                    ys,
                    orig_ys,
                    retracted: vec![false; n],
                    tree,
                };
                let resp = build_snapshot(&sess, n / 2, n_plot.unwrap_or(160), n_scan.unwrap_or(64));
                *slot = Some(sess);
                serde_json::to_string(&resp).unwrap()
            }
            Command::Upsert {
                t,
                y,
                query_t,
                n_plot,
                n_scan,
            } => {
                let Some(sess) = slot.as_mut() else {
                    return serde_json::json!({ "ok": false, "error": "not initialized" }).to_string();
                };
                let idx = t.min(sess.n.saturating_sub(1));
                sess.ys[idx] = y;
                sess.retracted[idx] = false;
                let leaf = match KalmanFilter2D::from_constant_velocity(1.0, sess.q_accel, sess.r_obs, y) {
                    Ok(l) => l,
                    Err(e) => {
                        return serde_json::json!({ "ok": false, "error": e.to_string() }).to_string();
                    }
                };
                sess.tree.upsert(EventKey::at(idx as i64), leaf);
                let resp = build_snapshot(
                    sess,
                    query_t.unwrap_or(idx),
                    n_plot.unwrap_or(160),
                    n_scan.unwrap_or(64),
                );
                serde_json::to_string(&resp).unwrap()
            }
            Command::Retract {
                t,
                query_t,
                n_plot,
                n_scan,
            } => {
                let Some(sess) = slot.as_mut() else {
                    return serde_json::json!({ "ok": false, "error": "not initialized" }).to_string();
                };
                let idx = t.min(sess.n.saturating_sub(1));
                sess.retracted[idx] = true;
                // Replace with missing-observation transition step so dt continuity is preserved
                let q = sess.q_accel;
                let f_mat = [1.0, 1.0, 0.0, 1.0];
                let q_mat = [q / 3.0, q / 2.0, q / 2.0, q];
                let miss = KalmanFilter2D::from_missing(f_mat, q_mat).unwrap();
                sess.tree.upsert(EventKey::at(idx as i64), miss);
                let resp = build_snapshot(
                    sess,
                    query_t.unwrap_or(idx),
                    n_plot.unwrap_or(160),
                    n_scan.unwrap_or(64),
                );
                serde_json::to_string(&resp).unwrap()
            }
            Command::Restore {
                t,
                query_t,
                n_plot,
                n_scan,
            } => {
                let Some(sess) = slot.as_mut() else {
                    return serde_json::json!({ "ok": false, "error": "not initialized" }).to_string();
                };
                let idx = t.min(sess.n.saturating_sub(1));
                let orig = sess.orig_ys[idx];
                sess.ys[idx] = orig;
                sess.retracted[idx] = false;
                let leaf = KalmanFilter2D::from_constant_velocity(1.0, sess.q_accel, sess.r_obs, orig)
                    .unwrap();
                sess.tree.upsert(EventKey::at(idx as i64), leaf);
                let resp = build_snapshot(
                    sess,
                    query_t.unwrap_or(idx),
                    n_plot.unwrap_or(160),
                    n_scan.unwrap_or(64),
                );
                serde_json::to_string(&resp).unwrap()
            }
            Command::BenchEdit {
                t,
                y_alt,
                query_t,
                reps,
            } => {
                let Some(sess) = slot.as_mut() else {
                    return serde_json::json!({ "ok": false, "error": "not initialized" }).to_string();
                };
                let idx = t.min(sess.n.saturating_sub(1));
                let qt = query_t.min(sess.n.saturating_sub(1));
                let orig = sess.ys[idx];
                let reps = reps.clamp(1, 5000);
                let mut acc = 0.0f64;
                for i in 0..reps {
                    let y = if i % 2 == 0 { y_alt } else { orig };
                    let leaf = KalmanFilter2D::from_constant_velocity(1.0, sess.q_accel, sess.r_obs, y)
                        .unwrap();
                    sess.tree.upsert(EventKey::at(idx as i64), leaf);
                    let (sm, _) = smooth_2d(&sess.tree, qt as i64, sess.m0, sess.p0).unwrap();
                    let ll = sess
                        .tree
                        .summary()
                        .log_marginal_likelihood(sess.m0, sess.p0)
                        .unwrap();
                    acc += sm[0] + ll * 1e-6;
                }
                // Ensure final state is restored to `orig`
                let leaf = KalmanFilter2D::from_constant_velocity(1.0, sess.q_accel, sess.r_obs, orig)
                    .unwrap();
                sess.tree.upsert(EventKey::at(idx as i64), leaf);
                let (sm, sp) = smooth_2d(&sess.tree, qt as i64, sess.m0, sess.p0).unwrap();
                let ll = sess
                    .tree
                    .summary()
                    .log_marginal_likelihood(sess.m0, sess.p0)
                    .unwrap();
                serde_json::json!({
                    "ok": true,
                    "reps": reps,
                    "checksum": acc,
                    "smooth_mean": sm,
                    "smooth_cov": sp,
                    "log_marginal_likelihood": ll
                })
                .to_string()
            }
            Command::Digest { top_k, threshold } => {
                let Some(sess) = slot.as_ref() else {
                    return serde_json::json!({ "ok": false, "error": "not initialized" }).to_string();
                };
                let dig = build_stream_digest(
                    "wasm_live_stream",
                    &sess.ts,
                    &sess.ys,
                    top_k.unwrap_or(4),
                    threshold.unwrap_or(12.0),
                )
                .unwrap();
                serde_json::json!({
                    "ok": true,
                    "n_ticks": dig.n_ticks,
                    "raw_approx_tokens": dig.raw_approx_tokens,
                    "digest_approx_tokens": dig.digest_approx_tokens,
                    "compression_ratio": dig.compression_ratio,
                    "first_alarm_ts": dig.first_alarm_ts,
                    "anytime_p_value": dig.anytime_p_value,
                    "markdown": dig.to_markdown()
                })
                .to_string()
            }
            Command::GetRaw => {
                let Some(sess) = slot.as_ref() else {
                    return serde_json::json!({ "ok": false, "error": "not initialized" }).to_string();
                };
                serde_json::json!({
                    "ok": true,
                    "q_accel": sess.q_accel,
                    "r_obs": sess.r_obs,
                    "m0": sess.m0,
                    "p0": sess.p0,
                    "ys": sess.ys,
                    "retracted": sess.retracted
                })
                .to_string()
            }
        }
    })
}

/// Allocate `len` bytes in WASM linear memory for JS -> Rust UTF-8 string passing.
#[no_mangle]
pub extern "C" fn bf_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free `len` bytes previously allocated by `bf_alloc` or returned by `bf_exec`.
///
/// # Safety
/// `ptr` must have been allocated by `bf_alloc` or `bf_exec` with exact capacity `len`.
#[no_mangle]
pub unsafe extern "C" fn bf_dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// Execute a JSON command stored at `(ptr, len)`.
/// Returns a packed `u64`: lower 32 bits = response pointer, upper 32 bits = response byte length.
///
/// # Safety
/// `(ptr, len)` must point to valid UTF-8 bytes in WASM linear memory.
#[no_mangle]
pub unsafe extern "C" fn bf_exec(ptr: *const u8, len: usize) -> u64 {
    let slice = std::slice::from_raw_parts(ptr, len);
    let input = std::str::from_utf8(slice).unwrap_or("{}");
    let mut resp_bytes = exec_json(input).into_bytes();
    resp_bytes.shrink_to_fit();
    let out_len = resp_bytes.len();
    let out_ptr = resp_bytes.as_mut_ptr();
    std::mem::forget(resp_bytes);
    ((out_len as u64) << 32) | (out_ptr as usize as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_session_roundtrip_init_upsert_retract_digest() {
        let init_out: serde_json::Value = serde_json::from_str(&exec_json(
            r#"{"cmd":"init","n":1000,"seed":42,"shock_t":500,"shock_step":8.0,"shock_vel":0.01}"#,
        ))
        .unwrap();
        assert_eq!(init_out["ok"], true);
        let best_cp = init_out["best_changepoint_t"].as_u64().unwrap();
        assert!(
            (470..=530).contains(&best_cp),
            "expected shock localization near 500, got {best_cp}"
        );
        assert!(init_out["best_log_bf"].as_f64().unwrap() > 15.0);

        let up_out: serde_json::Value = serde_json::from_str(&exec_json(
            r#"{"cmd":"upsert","t":250,"y":25.0,"query_t":250}"#,
        ))
        .unwrap();
        assert_eq!(up_out["ok"], true);

        let ret_out: serde_json::Value = serde_json::from_str(&exec_json(
            r#"{"cmd":"retract","t":250,"query_t":250}"#,
        ))
        .unwrap();
        assert_eq!(ret_out["ok"], true);

        let dig_out: serde_json::Value =
            serde_json::from_str(&exec_json(r#"{"cmd":"digest"}"#)).unwrap();
        assert_eq!(dig_out["ok"], true);
        assert!(dig_out["markdown"].as_str().unwrap().contains("Stream Digest"));
    }
}
