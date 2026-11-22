use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn cli_and_mcp_drill_preserve_parsed_values() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/drill_precision.csv"
    );
    let binary = env!("CARGO_BIN_EXE_backfold");
    let cli = Command::new(binary)
        .args(["drill", path, "--start", "10", "--end", "13"])
        .output()
        .unwrap();
    assert!(cli.status.success());
    let cli_text = String::from_utf8(cli.stdout).unwrap();

    let mut server = Command::new(binary)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let request = json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "backfold_drill_csv", "arguments": {"path": path, "start": 10, "end": 13}}
    });
    writeln!(server.stdin.take().unwrap(), "{request}").unwrap();
    let reply = server.wait_with_output().unwrap();
    assert!(reply.status.success());
    let rpc: Value = serde_json::from_slice(&reply.stdout).unwrap();
    assert_eq!(rpc["result"]["isError"], false);
    let mcp_text = rpc["result"]["content"][0]["text"].as_str().unwrap();

    fn ticks(text: &str) -> Vec<(i64, u64)> {
        text.lines()
            .filter_map(|line| {
                let (time, value) = line.split_once(',')?;
                Some((time.parse().ok()?, value.parse::<f64>().unwrap().to_bits()))
            })
            .collect()
    }
    let expected = ticks(include_str!("fixtures/drill_precision.csv"));
    assert_eq!(ticks(&cli_text), expected);
    assert_eq!(ticks(mcp_text), expected);
}
