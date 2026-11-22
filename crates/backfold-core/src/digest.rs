//! Top-down O(K log T) lossless-recoverable `<350`-token causal stream digest for AI agents.
//!
//! Instead of dumping raw telemetry rows into an LLM context window (blowing tokens and
//! obscuring rare bursts) or averaging them with `describe()` (destroying changepoints),
//! `build_stream_digest` inspects the already-computed `FoldTree` monoids top-down to surface:
//! - Robust warm-up scale (median + MAD) so 3x-louder distractor noise never causes false flags.
//! - Exact Page CUSUM changepoint onset, peak, and O(log T) first-crossing alarm timestamp.
//! - Anytime-valid Shin-Ramdas-Rinaldo e-CUSUM detector and Ville p-value.
//! - Exact Särkkä-García-Fernández Kalman level & velocity estimate at stream end.
//! - Top-K highest-surprise subintervals with exact `[t_start, t_end]` drilldown cursors.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::key::EventKey;
use crate::monoid::{BackfoldError, check_all_finite};
use crate::ops::{CusumMonoid, EProcessMonoid, KalmanFilter1D, MomentsMonoid};
use crate::tree::FoldTree;

/// One high-surprise subinterval surfaced by top-down monoid descent.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DigestWindow {
    pub t_start: i64,
    pub t_end: i64,
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub t_min: i64,
    pub t_max: i64,
    pub surprise_z: f64,
    pub kind: String,
}

/// Compact, strictly causal, lossless-recoverable digest of a univariate stream.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StreamDigest {
    pub channel: String,
    pub n_ticks: u64,
    pub t_first: i64,
    pub t_last: i64,
    pub raw_approx_tokens: usize,
    pub digest_approx_tokens: usize,
    pub compression_ratio: f64,
    pub baseline_median: f64,
    pub baseline_mad_std: f64,
    pub global_mean: f64,
    pub global_std: f64,
    pub global_min: f64,
    pub global_min_ts: i64,
    pub global_max: f64,
    pub global_max_ts: i64,
    pub cusum_peak_pos: f64,
    pub cusum_peak_neg: f64,
    pub cusum_onset_ts: Option<i64>,
    pub cusum_peak_ts: Option<i64>,
    pub first_alarm_ts: Option<i64>,
    pub e_detector: f64,
    pub e_detector_peak_valid: f64,
    pub anytime_p_value: f64,
    pub kalman_level: f64,
    pub kalman_std: f64,
    pub hotspots: Vec<DigestWindow>,
}

/// Robust median and MAD-scaled sigma (1.4826 * MAD) over a causal warm-up prefix.
pub fn causal_robust_scale(prefix: &[f64]) -> (f64, f64) {
    if prefix.is_empty() {
        return (0.0, 1.0);
    }
    let mut sorted: Vec<f64> = prefix.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return (0.0, 1.0);
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    let med = if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    };
    let mut devs: Vec<f64> = sorted.iter().map(|&v| (v - med).abs()).collect();
    devs.sort_by(|a, b| a.total_cmp(b));
    let mad = if devs.len() % 2 == 1 {
        devs[mid]
    } else {
        0.5 * (devs[mid - 1] + devs[mid])
    };
    let sigma = (mad * 1.4826).max(1e-6);
    (med, sigma)
}

/// Build a lossless-recoverable causal digest from timestamped observations.
///
/// Uses only the first `warmup_len` observations (default `min(n/10, 256).max(16)`) to
/// calibrate robust `(median, mad_sigma)`, then builds `MomentsMonoid`, `CusumMonoid`,
/// `EProcessMonoid`, and `KalmanFilter1D` trees in O(T) and walks the moments tree
/// top-down in O(K log T) to surface rare acute bursts and regime shifts.
pub fn build_stream_digest(
    channel: &str,
    timestamps: &[i64],
    values: &[f64],
    top_k: usize,
    cusum_threshold: f64,
) -> Result<StreamDigest, BackfoldError> {
    if timestamps.len() != values.len() {
        return Err(BackfoldError::DimensionMismatch {
            expected: timestamps.len(),
            got: values.len(),
        });
    }
    if timestamps.is_empty() {
        return Err(BackfoldError::InvalidParameter("empty stream"));
    }
    check_all_finite(values, "values")?;

    let n = values.len();
    let warmup_len = (n / 10).clamp(16, 256).min(n);
    let (med, mad_std) = causal_robust_scale(&values[..warmup_len]);
    let ep_margin = (2.0 / (warmup_len as f64).sqrt()).clamp(0.05, 0.25);
    let sorted = timestamps.windows(2).all(|w| w[0] <= w[1]);

    let mut mom_entries = Vec::with_capacity(n);
    let mut cus_entries = Vec::with_capacity(n);
    let mut ep_entries = Vec::with_capacity(n);
    let mut kf_entries = Vec::with_capacity(n);

    for i in 0..n {
        let ts = timestamps[i];
        let x = values[i];
        let z = (x - med) / mad_std;
        let key = EventKey::at(ts);
        mom_entries.push((key, MomentsMonoid::singleton(ts, x)?));
        cus_entries.push((key, CusumMonoid::singleton_z(ts, z, 0.5)?));
        ep_entries.push((key, EProcessMonoid::singleton_z_composite(ts, z, ep_margin)?));
        let dt = if i == 0 {
            1.0
        } else {
            ((timestamps[i] - timestamps[i - 1]).max(1)) as f64
        };
        kf_entries.push((
            key,
            KalmanFilter1D::from_random_walk(dt, 0.05 * mad_std * mad_std, mad_std * mad_std, x)?,
        ));
    }

    let (mom_tree, cus_tree, ep_tree, kf_tree): (
        FoldTree<MomentsMonoid, 16>,
        FoldTree<CusumMonoid, 16>,
        FoldTree<EProcessMonoid, 16>,
        FoldTree<KalmanFilter1D, 16>,
    ) = if sorted {
        (
            FoldTree::from_sorted((), mom_entries),
            FoldTree::from_sorted((), cus_entries),
            FoldTree::from_sorted((), ep_entries),
            FoldTree::from_sorted((), kf_entries),
        )
    } else {
        (
            FoldTree::from_unsorted((), mom_entries),
            FoldTree::from_unsorted((), cus_entries),
            FoldTree::from_unsorted((), ep_entries),
            FoldTree::from_unsorted((), kf_entries),
        )
    };

    let g_mom = mom_tree.summary();
    let g_cus = cus_tree.summary();
    let g_ep = ep_tree.summary();
    let (kf_mean, kf_var) = kf_tree.summary().posterior(med, 4.0 * mad_std * mad_std)?;

    let first_alarm_ts = cus_tree
        .find_first_prefix(|m| m.max_peak() >= cusum_threshold)
        .map(|(k, _)| k.event_time);

    // Top-down O(K log T) search for highest-surprise subintervals (captures acute spikes & regime shifts)
    let depth = mom_tree.depth();
    let target_depth = depth.saturating_sub(1).max(1);
    let raw_hot = mom_tree.top_k_intervals(top_k.max(1), target_depth, |m| {
        if m.n == 0 {
            return 0.0;
        }
        let max_dev = (m.max - med).abs().max((m.min - med).abs()) / mad_std;
        let mean_shift = (m.mean - med).abs() / mad_std;
        let vol_ratio = m.std() / mad_std;
        // Blend extreme burst sensitivity with sustained shift & volatility spike
        0.55 * max_dev + 0.35 * mean_shift + 0.10 * vol_ratio
    });

    let hotspots: Vec<DigestWindow> = raw_hot
        .into_iter()
        .map(|(k0, k1, cnt, score, m)| {
            let mean_shift = (m.mean - med).abs() / mad_std;
            let peak_dev = (m.max - med).abs().max((m.min - med).abs()) / mad_std;
            let kind = if peak_dev >= 4.5 && mean_shift < 0.5 * peak_dev {
                "acute_burst"
            } else if mean_shift >= 1.2 {
                "regime_shift"
            } else {
                "elevated_activity"
            };
            DigestWindow {
                t_start: k0.event_time,
                t_end: k1.event_time,
                count: cnt,
                mean: m.mean,
                std: m.std(),
                min: m.min,
                max: m.max,
                t_min: m.t_min,
                t_max: m.t_max,
                surprise_z: score,
                kind: kind.to_string(),
            }
        })
        .collect();

    let raw_approx_tokens = n * 6; // conservative ~6 tokens per CSV row "timestamp,value\n"
    let mut digest = StreamDigest {
        channel: channel.to_string(),
        n_ticks: g_mom.n,
        t_first: g_mom.t_first,
        t_last: g_mom.t_last,
        raw_approx_tokens,
        digest_approx_tokens: 0,
        compression_ratio: 1.0,
        baseline_median: med,
        baseline_mad_std: mad_std,
        global_mean: g_mom.mean,
        global_std: g_mom.std(),
        global_min: g_mom.min,
        global_min_ts: g_mom.t_min,
        global_max: g_mom.max,
        global_max_ts: g_mom.t_max,
        cusum_peak_pos: g_cus.peak_pos(),
        cusum_peak_neg: g_cus.peak_neg(),
        cusum_onset_ts: if g_cus.max_peak() >= cusum_threshold {
            Some(g_cus.onset_ts())
        } else {
            None
        },
        cusum_peak_ts: if g_cus.max_peak() >= cusum_threshold {
            Some(g_cus.peak_ts())
        } else {
            None
        },
        first_alarm_ts,
        e_detector: g_ep.e_detector(),
        e_detector_peak_valid: g_ep.peak_e_detector_valid(),
        anytime_p_value: g_ep.anytime_p_value(),
        kalman_level: kf_mean,
        kalman_std: kf_var.max(0.0).sqrt(),
        hotspots,
    };

    let md = digest.to_markdown();
    let approx_toks = md.len().div_ceil(4);
    digest.digest_approx_tokens = approx_toks;
    digest.compression_ratio = raw_approx_tokens as f64 / (approx_toks.max(1) as f64);
    Ok(digest)
}

impl StreamDigest {
    /// Render a compact, pure-Markdown digest designed for AI coding & monitoring agents (<350 tokens).
    pub fn to_markdown(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str(&format!(
            "# Stream Digest: `{}` ({} ticks, ts `{}..{}`)\n",
            self.channel, self.n_ticks, self.t_first, self.t_last
        ));
        out.push_str(&format!(
            "- **Baseline (causal warm-up):** median `{:.4}`, MAD-sigma `{:.4}`\n",
            self.baseline_median, self.baseline_mad_std
        ));
        out.push_str(&format!(
            "- **Global:** mean `{:.4}`, std `{:.4}`, min `{:.4}` @ `{}`, max `{:.4}` @ `{}`\n",
            self.global_mean,
            self.global_std,
            self.global_min,
            self.global_min_ts,
            self.global_max,
            self.global_max_ts
        ));
        out.push_str(&format!(
            "- **Kalman state (end):** level `{:.4} ± {:.4}`\n",
            self.kalman_level, self.kalman_std
        ));
        out.push_str(&format!(
            "- **CUSUM / Anytime E-Detector:** peak+ `{:.2}`, peak- `{:.2}`, first_alarm `{:?}`, onset `{:?}`, peak `{:?}`, e_det `{:.2e}` (p_anytime `{:.2e}`)\n",
            self.cusum_peak_pos,
            self.cusum_peak_neg,
            self.first_alarm_ts,
            self.cusum_onset_ts,
            self.cusum_peak_ts,
            self.e_detector_peak_valid,
            self.anytime_p_value
        ));
        if !self.hotspots.is_empty() {
            out.push_str("\n| Window `[ts]` | Kind | n | Mean | Peak `[min..max]` | Score |\n");
            out.push_str("|---|---|---:|---:|---|---:|\n");
            for h in &self.hotspots {
                out.push_str(&format!(
                    "| `{}..{}` | {} | {} | {:.3} | {:.3}@{}, {:.3}@{} | {:.2} |\n",
                    h.t_start, h.t_end, h.kind, h.count, h.mean, h.min, h.t_min, h.max, h.t_max, h.surprise_z
                ));
            }
        }
        out.push_str(
            "\n*Drilldown:* `backfold drill <file> --start <ts> --end <ts>` recovers exact raw ticks.\n",
        );
        out
    }
}
