use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

#[test]
fn mcp_server_end_to_end_all_five_tools() {
    // Write a deterministic multi-column CSV with an injected regime shock in ch0 + ch1
    let uniq = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let csv_path = std::env::temp_dir().join(format!("backfold_mcp_test_{uniq}.csv"));
    {
        let mut f = std::fs::File::create(&csv_path).unwrap();
        writeln!(f, "ts,sensor_a,sensor_b,sensor_c,sensor_d").unwrap();
        let mut seed = 1337u64;
        for t in 0..320 {
            let mut next_n = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                u * 2.0
            };
            let base = next_n();
            let mut a = base + 0.1 * next_n();
            let mut b = 0.8 * base + 0.2 * next_n();
            let c = next_n() * 1.5;
            let d = next_n() * 0.8;
            if t >= 180 {
                a += 4.5;
                b += 3.6;
            }
            writeln!(f, "{t},{a:.6},{b:.6},{c:.6},{d:.6}").unwrap();
        }
    }
    let path_str = csv_path.to_str().unwrap().to_string();

    let mut child = Command::new(env!("CARGO_BIN_EXE_backfold"))
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn backfold mcp");

    let mut stdin = child.stdin.take().unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());

    fn send_recv(
        stdin: &mut std::process::ChildStdin,
        reader: &mut BufReader<std::process::ChildStdout>,
        req: Value,
    ) -> Value {
        writeln!(stdin, "{req}").unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).expect("valid JSON-RPC response")
    }

    // 1. initialize
    let init = send_recv(
        &mut stdin,
        &mut reader,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": { "protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": { "name": "test", "version": "0.1" } }
        }),
    );
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["serverInfo"]["name"], "backfold");

    // 2. notification (must NOT produce a response)
    writeln!(
        stdin,
        "{}",
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })
    )
    .unwrap();
    stdin.flush().unwrap();

    // 3. ping
    let ping = send_recv(
        &mut stdin,
        &mut reader,
        json!({ "jsonrpc": "2.0", "id": 2, "method": "ping" }),
    );
    assert_eq!(ping["id"], 2);
    assert!(ping["result"].is_object());

    // 4. tools/list
    let list = send_recv(
        &mut stdin,
        &mut reader,
        json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/list" }),
    );
    let tools = list["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 5);

    // 5. backfold_digest_csv
    let dig = send_recv(
        &mut stdin,
        &mut reader,
        json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "backfold_digest_csv",
                "arguments": { "path": path_str, "time_col": "ts", "value_col": "sensor_a", "threshold": 10.0 }
            }
        }),
    );
    assert_eq!(dig["result"]["isError"], false);
    let dig_txt = dig["result"]["content"][0]["text"].as_str().unwrap();
    assert!(dig_txt.contains("# Stream Digest: `sensor_a`"));

    // 6. backfold_drill_csv
    let drill = send_recv(
        &mut stdin,
        &mut reader,
        json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "backfold_drill_csv",
                "arguments": { "path": path_str, "start": 175, "end": 185, "value_col": "sensor_a" }
            }
        }),
    );
    assert_eq!(drill["result"]["isError"], false);
    let drill_txt = drill["result"]["content"][0]["text"].as_str().unwrap();
    assert!(drill_txt.contains("n**: 11"));

    // 7. backfold_kalman_rts_csv
    let krts = send_recv(
        &mut stdin,
        &mut reader,
        json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "backfold_kalman_rts_csv",
                "arguments": { "path": path_str, "value_col": "sensor_a", "query_ts": [100, 180, 260] }
            }
        }),
    );
    assert_eq!(krts["result"]["isError"], false);
    let krts_json: Value =
        serde_json::from_str(krts["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(krts_json["rts_smoothed"].as_array().unwrap().len(), 3);
    let best_cp = krts_json["changepoint_scan"]["best"]["t"].as_i64().unwrap();
    assert!(
        (165..=195).contains(&best_cp),
        "expected shock changepoint near 180, got {best_cp}"
    );

    // 8. backfold_covariance_pca_csv
    let pca = send_recv(
        &mut stdin,
        &mut reader,
        json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "backfold_covariance_pca_csv",
                "arguments": { "path": path_str, "value_cols": ["sensor_a", "sensor_b", "sensor_c", "sensor_d"], "top_k": 2 }
            }
        }),
    );
    assert_eq!(pca["result"]["isError"], false);
    let pca_json: Value =
        serde_json::from_str(pca["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(pca_json["eigenvalues"].as_array().unwrap().len(), 2);
    assert!(pca_json["explained_variance_ratio"][0].as_f64().unwrap() > 0.6);

    // 9. backfold_ebh_sift_csv
    let sift = send_recv(
        &mut stdin,
        &mut reader,
        json!({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "backfold_ebh_sift_csv",
                "arguments": { "path": path_str, "warmup": 120, "alpha": 0.05 }
            }
        }),
    );
    assert_eq!(sift["result"]["isError"], false);
    let sift_json: Value =
        serde_json::from_str(sift["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
    let rej_cols: Vec<&str> = sift_json["rejected_channels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["column"].as_str().unwrap())
        .collect();
    assert_eq!(rej_cols, vec!["sensor_a", "sensor_b"]);

    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success());
    let _ = std::fs::remove_file(csv_path);
}
