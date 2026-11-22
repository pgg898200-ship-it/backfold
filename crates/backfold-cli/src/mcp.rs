//! Zero-dependency Model Context Protocol (MCP 2024-11-05) stdio JSON-RPC 2.0 server.
//!
//! Exposes 5 streaming statistical inference & compression tools to any MCP-compatible AI host
//! (Claude Desktop, Cursor, Windsurf, Claude Code, custom agents):
//! - `backfold_digest_csv`
//! - `backfold_drill_csv`
//! - `backfold_kalman_rts_csv`
//! - `backfold_covariance_pca_csv`
//! - `backfold_ebh_sift_csv`

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

use backfold_core::digest::build_stream_digest;
use backfold_core::ops::{
    changepoint_shock_bayes_factor_2d, sift_streams_fdr, smooth_2d, CovarianceMonoid,
    KalmanFilter2D, MomentsMonoid,
};
use backfold_core::{EventKey, FoldTree};
use serde_json::{json, Value};

const PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug)]
struct CsvTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn read_csv_table(path: &str) -> Result<CsvTable, String> {
    let f = File::open(path).map_err(|e| format!("failed to open '{path}': {e}"))?;
    let reader = BufReader::new(f);
    let mut raw_rows: Vec<Vec<String>> = Vec::new();
    for line_res in reader.lines() {
        let line = line_res.map_err(|e| format!("read error: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cols: Vec<String> = trimmed.split(',').map(|s| s.trim().to_string()).collect();
        raw_rows.push(cols);
    }
    if raw_rows.is_empty() {
        return Err(format!("CSV '{path}' is empty"));
    }
    // Detect whether row 0 is a header (any cell fails f64 parse)
    let has_header = raw_rows[0].iter().any(|c| c.parse::<f64>().is_err());
    if has_header {
        let headers = raw_rows.remove(0);
        Ok(CsvTable {
            headers,
            rows: raw_rows,
        })
    } else {
        let ncols = raw_rows[0].len();
        let headers = (0..ncols).map(|i| format!("col_{i}")).collect();
        Ok(CsvTable {
            headers,
            rows: raw_rows,
        })
    }
}

fn resolve_col(table: &CsvTable, spec: Option<&Value>, default_idx: usize) -> Result<usize, String> {
    match spec {
        None | Some(Value::Null) => {
            if default_idx < table.headers.len() {
                Ok(default_idx)
            } else {
                Err(format!(
                    "default column index {default_idx} out of bounds (table has {} columns)",
                    table.headers.len()
                ))
            }
        }
        Some(Value::Number(n)) => {
            let idx = n
                .as_u64()
                .ok_or_else(|| format!("column index must be non-negative integer, got {n}"))?
                as usize;
            if idx < table.headers.len() {
                Ok(idx)
            } else {
                Err(format!(
                    "column index {idx} out of bounds (table has {} columns)",
                    table.headers.len()
                ))
            }
        }
        Some(Value::String(name)) => {
            if let Some(pos) = table.headers.iter().position(|h| h == name) {
                Ok(pos)
            } else if let Ok(idx) = name.parse::<usize>() {
                if idx < table.headers.len() {
                    Ok(idx)
                } else {
                    Err(format!("column index {idx} out of bounds"))
                }
            } else {
                Err(format!(
                    "column '{name}' not found; available headers: {:?}",
                    table.headers
                ))
            }
        }
        Some(other) => Err(format!("invalid column specifier: {other}")),
    }
}

fn extract_ts_and_val(
    table: &CsvTable,
    time_col: usize,
    value_col: usize,
) -> Result<(Vec<i64>, Vec<f64>), String> {
    let mut ts = Vec::with_capacity(table.rows.len());
    let mut vals = Vec::with_capacity(table.rows.len());
    let req = time_col.max(value_col);
    for (idx, row) in table.rows.iter().enumerate() {
        if row.len() <= req {
            continue;
        }
        let t = row[time_col]
            .parse::<i64>()
            .or_else(|_| row[time_col].parse::<f64>().map(|v| v.round() as i64))
            .map_err(|e| format!("row {idx} bad timestamp '{}': {e}", row[time_col]))?;
        let v = row[value_col]
            .parse::<f64>()
            .map_err(|e| format!("row {idx} bad value '{}': {e}", row[value_col]))?;
        if v.is_finite() {
            ts.push(t);
            vals.push(v);
        }
    }
    if ts.is_empty() {
        return Err("no valid finite numeric rows found".into());
    }
    Ok((ts, vals))
}

fn tools_schema() -> Value {
    json!({
        "tools": [
            {
                "name": "backfold_digest_csv",
                "description": "Compress an arbitrarily long CSV time series into a <300-token lossless-recoverable Markdown statistical digest with exact CUSUM alarm timestamps, anytime-valid e-process p-value, Kalman level/std, and ranked hotspot windows for drilldown.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to local CSV file" },
                        "time_col": { "description": "Timestamp column index (int) or header name (string). Default: 0" },
                        "value_col": { "description": "Value column index (int) or header name (string). Default: 1" },
                        "top_k": { "type": "integer", "description": "Number of top hotspot intervals to include. Default: 5" },
                        "threshold": { "type": "number", "description": "CUSUM alarm threshold in sigma units. Default: 10.0" }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "backfold_drill_csv",
                "description": "Lossless drilldown into a specific timestamp window [start, end] of a CSV time series, returning exact 106-bit compensated moments (mean, std, skewness, kurtosis, min/max) and raw ticks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to local CSV file" },
                        "start": { "type": "integer", "description": "Inclusive start timestamp" },
                        "end": { "type": "integer", "description": "Inclusive end timestamp" },
                        "time_col": { "description": "Timestamp column index or header name. Default: 0" },
                        "value_col": { "description": "Value column index or header name. Default: 1" },
                        "max_rows": { "type": "integer", "description": "Maximum raw rows returned to avoid context bloat. Default: 120" }
                    },
                    "required": ["path", "start", "end"]
                }
            },
            {
                "name": "backfold_kalman_rts_csv",
                "description": "Run exact O(log T) 2D constant-velocity Kalman filtering, Rauch-Tung-Striebel (RTS) smoothing at requested timestamps, exact marginal log-likelihood log p(y), and translation-invariant Bayesian changepoint shock localization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to local CSV file" },
                        "time_col": { "description": "Timestamp column index or header name. Default: 0" },
                        "value_col": { "description": "Value column index or header name. Default: 1" },
                        "query_ts": { "type": "array", "items": { "type": "integer" }, "description": "Optional timestamps for exact RTS smoothed state queries" },
                        "q_accel": { "type": "number", "description": "Process acceleration noise variance. Default: 0.001" },
                        "r_obs": { "type": "number", "description": "Observation noise variance. Default: 0.25" },
                        "scan_changepoints": { "type": "boolean", "description": "Run O(K log T) Bayesian shock changepoint scan. Default: true" },
                        "n_scan_points": { "type": "integer", "description": "Number of candidate split timestamps to scan. Default: 48" }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "backfold_covariance_pca_csv",
                "description": "Compute exact 106-bit Knuth-compensated multivariate covariance, correlation, cyclic-Jacobi streaming PCA (eigenvalues, explained variance ratios, principal components), and top Mahalanobis outliers over a CSV or [start, end] time range.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to local CSV file" },
                        "value_cols": { "type": "array", "description": "List of >= 2 column indices or header names" },
                        "time_col": { "description": "Timestamp column index or header name. Default: 0" },
                        "start": { "type": "integer", "description": "Optional inclusive start timestamp filter" },
                        "end": { "type": "integer", "description": "Optional inclusive end timestamp filter" },
                        "top_k": { "type": "integer", "description": "Optional number of top principal components to return" }
                    },
                    "required": ["path", "value_cols"]
                }
            },
            {
                "name": "backfold_ebh_sift_csv",
                "description": "Run Wang-Ramdas arbitrary-dependence e-BH FDR-controlled multi-stream anomaly sifting across multiple CSV channels (provable FDR <= alpha under heavy-tailed or correlated noise).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to local CSV file" },
                        "value_cols": { "type": "array", "description": "Optional list of channel column indices/names (defaults to all columns except time_col)" },
                        "time_col": { "description": "Timestamp column index or header name. Default: 0" },
                        "warmup": { "type": "integer", "description": "Strictly causal baseline warmup length. Default: 100" },
                        "alpha": { "type": "number", "description": "Target FDR level in (0, 1). Default: 0.05" },
                        "null_margin": { "type": "number", "description": "Composite-null tolerance in [0, 0.5]. Default: 0.15" }
                    },
                    "required": ["path"]
                }
            }
        ]
    })
}

fn call_digest_csv(args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing required string argument 'path'")?;
    let table = read_csv_table(path)?;
    let time_col = resolve_col(&table, args.get("time_col"), 0)?;
    let value_col = resolve_col(&table, args.get("value_col"), 1)?;
    let top_k = args.get("top_k").and_then(Value::as_u64).unwrap_or(5) as usize;
    let threshold = args
        .get("threshold")
        .and_then(Value::as_f64)
        .unwrap_or(10.0);
    let (ts, vals) = extract_ts_and_val(&table, time_col, value_col)?;
    let channel = table
        .headers
        .get(value_col)
        .cloned()
        .unwrap_or_else(|| path.to_string());
    let dig = build_stream_digest(&channel, &ts, &vals, top_k, threshold)
        .map_err(|e| format!("digest error: {e}"))?;
    Ok(dig.to_markdown())
}

fn call_drill_csv(args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing required string argument 'path'")?;
    let start = args
        .get("start")
        .and_then(Value::as_i64)
        .ok_or("missing required integer argument 'start'")?;
    let end = args
        .get("end")
        .and_then(Value::as_i64)
        .ok_or("missing required integer argument 'end'")?;
    let max_rows = args
        .get("max_rows")
        .and_then(Value::as_u64)
        .unwrap_or(120) as usize;
    let table = read_csv_table(path)?;
    let time_col = resolve_col(&table, args.get("time_col"), 0)?;
    let value_col = resolve_col(&table, args.get("value_col"), 1)?;
    let (ts, vals) = extract_ts_and_val(&table, time_col, value_col)?;
    let entries: Vec<_> = ts
        .iter()
        .zip(vals.iter())
        .map(|(&t, &x)| {
            MomentsMonoid::singleton(t, x)
                .map(|m| (EventKey::at(t), m))
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    let tree: FoldTree<MomentsMonoid, 4> = FoldTree::from_unsorted((), entries);
    let sub = tree.query_time_range(start, end);
    let rows = tree.iter_range(EventKey::time_start(start), EventKey::time_end(end));
    let mut out = format!(
        "# Drilldown `{}` [{}..{}]\n- **n**: {} | **mean**: {:.6} | **std**: {:.6} | **skew**: {:.4} | **kurt**: {:.4}\n- **min**: {:.6} @ t={} | **max**: {:.6} @ t={}\n\n```csv\ntimestamp,value\n",
        path,
        start,
        end,
        sub.n,
        sub.mean,
        sub.std(),
        sub.skewness(),
        sub.excess_kurtosis(),
        sub.min,
        sub.t_min,
        sub.max,
        sub.t_max
    );
    let total = rows.len();
    for (k, m) in rows.into_iter().take(max_rows) {
        out.push_str(&format!("{},{:?}\n", k.event_time, m.mean));
    }
    if total > max_rows {
        out.push_str(&format!(
            "# ... ({} more rows omitted; raise max_rows or narrow [start, end])\n",
            total - max_rows
        ));
    }
    out.push_str("```\n");
    Ok(out)
}

fn call_kalman_rts_csv(args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing required string argument 'path'")?;
    let q_accel = args
        .get("q_accel")
        .and_then(Value::as_f64)
        .unwrap_or(1e-3);
    let r_obs = args.get("r_obs").and_then(Value::as_f64).unwrap_or(0.25);
    let scan_cp = args
        .get("scan_changepoints")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let n_scan = args
        .get("n_scan_points")
        .and_then(Value::as_u64)
        .unwrap_or(48)
        .clamp(4, 256) as usize;

    let table = read_csv_table(path)?;
    let time_col = resolve_col(&table, args.get("time_col"), 0)?;
    let value_col = resolve_col(&table, args.get("value_col"), 1)?;
    let (ts, vals) = extract_ts_and_val(&table, time_col, value_col)?;

    let mut pairs: Vec<(i64, f64)> = ts.into_iter().zip(vals).collect();
    pairs.sort_by_key(|&(t, _)| t);

    let mut entries = Vec::with_capacity(pairs.len());
    for i in 0..pairs.len() {
        let dt = if i == 0 {
            1.0
        } else {
            ((pairs[i].0 - pairs[i - 1].0) as f64).max(1e-6)
        };
        let leaf = KalmanFilter2D::from_constant_velocity(dt, q_accel, r_obs, pairs[i].1)
            .map_err(|e| e.to_string())?;
        entries.push((EventKey::at(pairs[i].0), leaf));
    }
    let tree: FoldTree<KalmanFilter2D, 4> = FoldTree::from_sorted((), entries);
    let m0 = [pairs[0].1, 0.0];
    let p0 = [100.0, 0.0, 0.0, 10.0];
    let (filt_m, filt_p) = tree.summary().posterior(m0, p0).map_err(|e| e.to_string())?;
    let log_lik = tree
        .summary()
        .log_marginal_likelihood(m0, p0)
        .map_err(|e| e.to_string())?;

    let query_ts: Vec<i64> = if let Some(arr) = args.get("query_ts").and_then(Value::as_array) {
        arr.iter().filter_map(Value::as_i64).collect()
    } else {
        let n = pairs.len();
        [0, n / 4, n / 2, (3 * n) / 4, n.saturating_sub(1)]
            .into_iter()
            .map(|i| pairs[i].0)
            .collect()
    };

    let mut smoothed = Vec::with_capacity(query_ts.len());
    for t in query_ts {
        let (sm, sp) = smooth_2d(&tree, t, m0, p0).map_err(|e| e.to_string())?;
        smoothed.push(json!({
            "t": t,
            "level": sm[0],
            "velocity": sm[1],
            "level_std": sp[0].max(0.0).sqrt(),
            "velocity_std": sp[3].max(0.0).sqrt(),
        }));
    }

    let changepoint_scan = if scan_cp && pairs.len() >= 10 {
        let q_jump = [25.0 * r_obs.max(0.1), 0.0, 0.0, 1.0];
        let located = backfold_core::ops::locate_changepoint_2d(&tree, m0, p0, q_jump, n_scan)
            .map_err(|e| e.to_string())?;
        let n = pairs.len();
        let lo = (n / 20).max(2);
        let hi = n.saturating_sub(lo).max(lo + 1);
        let span = hi.saturating_sub(lo).max(1);
        let mut cands: Vec<(i64, f64)> = Vec::with_capacity(n_scan + 1);
        if let Some((bt, bbf)) = located {
            cands.push((bt, bbf));
        }
        for k in 0..n_scan {
            let idx = lo + (k * span) / n_scan.max(1);
            if idx < n {
                let t = pairs[idx].0;
                if located.is_some_and(|(bt, _)| bt == t) {
                    continue;
                }
                let bf = changepoint_shock_bayes_factor_2d(&tree, t, m0, p0, q_jump)
                    .map_err(|e| e.to_string())?;
                cands.push((t, bf));
            }
        }
        cands.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<Value> = cands
            .into_iter()
            .take(5)
            .map(|(t, bf)| json!({ "t": t, "log_bayes_factor": bf }))
            .collect();
        Some(json!({
            "best": top.first().cloned().unwrap_or(Value::Null),
            "top_candidates": top
        }))
    } else {
        None
    };

    let payload = json!({
        "n_ticks": pairs.len(),
        "t_first": pairs.first().map(|p| p.0),
        "t_last": pairs.last().map(|p| p.0),
        "filtered_end": {
            "level": filt_m[0],
            "velocity": filt_m[1],
            "level_std": filt_p[0].max(0.0).sqrt(),
            "velocity_std": filt_p[3].max(0.0).sqrt()
        },
        "log_marginal_likelihood": log_lik,
        "rts_smoothed": smoothed,
        "changepoint_scan": changepoint_scan
    });
    serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())
}

fn call_covariance_pca_csv(args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing required string argument 'path'")?;
    let col_specs = args
        .get("value_cols")
        .and_then(Value::as_array)
        .ok_or("missing required array argument 'value_cols'")?;
    if col_specs.len() < 2 {
        return Err("value_cols must contain at least 2 columns".into());
    }
    let table = read_csv_table(path)?;
    let time_col = resolve_col(&table, args.get("time_col"), 0)?;
    let mut val_cols = Vec::with_capacity(col_specs.len());
    for spec in col_specs {
        val_cols.push(resolve_col(&table, Some(spec), 0)?);
    }
    let dim = val_cols.len();
    let start_opt = args.get("start").and_then(Value::as_i64);
    let end_opt = args.get("end").and_then(Value::as_i64);
    let top_k = args
        .get("top_k")
        .and_then(Value::as_u64)
        .map(|v| v as usize);

    let mut entries = Vec::with_capacity(table.rows.len());
    let mut raw_ts = Vec::with_capacity(table.rows.len());
    let mut raw_flat = Vec::with_capacity(table.rows.len() * dim);

    for (r_idx, row) in table.rows.iter().enumerate() {
        let t = row
            .get(time_col)
            .ok_or_else(|| format!("row {r_idx} missing time_col {time_col}"))?
            .parse::<i64>()
            .or_else(|_| row[time_col].parse::<f64>().map(|v| v.round() as i64))
            .map_err(|e| format!("row {r_idx} bad timestamp: {e}"))?;
        let mut vec_x = Vec::with_capacity(dim);
        let mut ok = true;
        for &c in &val_cols {
            let Some(cell) = row.get(c) else {
                ok = false;
                break;
            };
            let v = cell
                .parse::<f64>()
                .map_err(|e| format!("row {r_idx} col {c} bad float '{cell}': {e}"))?;
            if !v.is_finite() {
                ok = false;
                break;
            }
            vec_x.push(v);
        }
        if !ok {
            continue;
        }
        let m = CovarianceMonoid::singleton(t, &vec_x).map_err(|e| e.to_string())?;
        entries.push((EventKey::at(t), m));
        if start_opt.is_none_or(|s| t >= s) && end_opt.is_none_or(|e| t <= e) {
            raw_ts.push(t);
            raw_flat.extend_from_slice(&vec_x);
        }
    }
    if entries.is_empty() {
        return Err("no valid rows parsed".into());
    }
    let tree: FoldTree<CovarianceMonoid, 4> = FoldTree::from_unsorted(dim, entries);
    let summary = match (start_opt, end_opt) {
        (Some(s), Some(e)) => tree.range(EventKey::time_start(s), EventKey::time_end(e)),
        (Some(s), None) => tree.suffix(EventKey::time_start(s)),
        (None, Some(e)) => tree.prefix(EventKey::time_end(e)),
        (None, None) => tree.summary(),
    };
    let pca = summary.pca(1, top_k);
    let corr_flat = summary.correlation();
    let corr_mat: Vec<Vec<f64>> = corr_flat.chunks_exact(dim).map(|r| r.to_vec()).collect();
    let comp_mat: Vec<Vec<f64>> = pca.components.chunks_exact(dim).map(|r| r.to_vec()).collect();

    let d2 = summary
        .mahalanobis_sq_batch(&raw_flat, 1e-6)
        .map_err(|e| e.to_string())?;
    let mut ranked: Vec<(i64, f64)> = raw_ts.into_iter().zip(d2).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top_outliers: Vec<Value> = ranked
        .into_iter()
        .take(5)
        .map(|(t, dist_sq)| json!({ "t": t, "mahalanobis_sq": dist_sq }))
        .collect();

    let col_names: Vec<String> = val_cols
        .iter()
        .map(|&c| table.headers[c].clone())
        .collect();
    let payload = json!({
        "n": summary.n,
        "columns": col_names,
        "mean": summary.mean(),
        "eigenvalues": pca.eigenvalues,
        "explained_variance_ratio": pca.explained_variance_ratio,
        "components": comp_mat,
        "correlation": corr_mat,
        "top_mahalanobis_outliers": top_outliers
    });
    serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())
}

fn call_ebh_sift_csv(args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing required string argument 'path'")?;
    let warmup = args.get("warmup").and_then(Value::as_u64).unwrap_or(100) as usize;
    let alpha = args.get("alpha").and_then(Value::as_f64).unwrap_or(0.05);
    let null_margin = args
        .get("null_margin")
        .and_then(Value::as_f64)
        .unwrap_or(0.15);

    let table = read_csv_table(path)?;
    let time_col = resolve_col(&table, args.get("time_col"), 0)?;
    let val_cols: Vec<usize> = if let Some(arr) = args.get("value_cols").and_then(Value::as_array) {
        let mut cols = Vec::with_capacity(arr.len());
        for spec in arr {
            cols.push(resolve_col(&table, Some(spec), 0)?);
        }
        cols
    } else {
        (0..table.headers.len()).filter(|&i| i != time_col).collect()
    };
    if val_cols.is_empty() {
        return Err("no value columns selected".into());
    }
    let n_streams = val_cols.len();
    let t_len = table.rows.len();
    if t_len <= warmup {
        return Err(format!(
            "need more rows ({t_len}) than warmup ({warmup})"
        ));
    }
    let mut flat = vec![0.0f64; n_streams * t_len];
    for (t, row) in table.rows.iter().enumerate() {
        for (ch, &c) in val_cols.iter().enumerate() {
            let cell = row
                .get(c)
                .ok_or_else(|| format!("row {t} missing column {c}"))?;
            let v = cell
                .parse::<f64>()
                .map_err(|e| format!("row {t} col {c} bad float '{cell}': {e}"))?;
            if !v.is_finite() {
                return Err(format!("non-finite float at row {t} col {c}"));
            }
            flat[ch * t_len + t] = v;
        }
    }
    let res = sift_streams_fdr(&flat, n_streams, t_len, warmup, alpha, null_margin)
        .map_err(|e| e.to_string())?;
    let rejected: Vec<Value> = res
        .rejected_indices
        .iter()
        .map(|&ch| {
            json!({
                "index": ch,
                "column": table.headers[val_cols[ch]],
                "e_value": res.e_values[ch],
                "baseline_median": res.baselines[ch].0,
                "baseline_mad_std": res.baselines[ch].1
            })
        })
        .collect();
    let payload = json!({
        "n_streams": n_streams,
        "n_ticks": t_len,
        "warmup": warmup,
        "alpha": alpha,
        "n_rejected": rejected.len(),
        "rejected_channels": rejected
    });
    serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())
}

fn dispatch_tool(name: &str, args: &Value) -> Result<String, String> {
    match name {
        "backfold_digest_csv" => call_digest_csv(args),
        "backfold_drill_csv" => call_drill_csv(args),
        "backfold_kalman_rts_csv" => call_kalman_rts_csv(args),
        "backfold_covariance_pca_csv" => call_covariance_pca_csv(args),
        "backfold_ebh_sift_csv" => call_ebh_sift_csv(args),
        other => Err(format!("unknown tool: {other}")),
    }
}

/// Run the newline-delimited JSON-RPC 2.0 MCP server over stdin/stdout until EOF.
pub fn run_mcp_server() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line_res in stdin.lock().lines() {
        let Ok(line) = line_res else {
            break;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") }
                });
                let _ = writeln!(out, "{err_resp}");
                let _ = out.flush();
                continue;
            }
        };

        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);

        // JSON-RPC notifications (no `id`) receive no response.
        let Some(req_id) = id else {
            continue;
        };

        let response = match method {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": "backfold",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }
            }),
            "ping" => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": tools_schema()
            }),
            "tools/call" => {
                let tool_name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let tool_args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match dispatch_tool(tool_name, &tool_args) {
                    Ok(text) => json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{ "type": "text", "text": text }],
                            "isError": false
                        }
                    }),
                    Err(msg) => json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{ "type": "text", "text": format!("error: {msg}") }],
                            "isError": true
                        }
                    }),
                }
            }
            other => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": { "code": -32601, "message": format!("method not found: {other}") }
            }),
        };

        if writeln!(out, "{response}").is_err() || out.flush().is_err() {
            break;
        }
    }
}
