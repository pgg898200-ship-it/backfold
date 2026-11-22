use std::process::Command;

#[test]
fn digest_and_drill_report_bad_flag_values_before_opening_the_csv() {
    for (command, flags, message) in [
        ("digest", &["--top-k"][..], "--top-k needs a value"),
        ("drill", &["--end"][..], "--end needs a value"),
        (
            "digest",
            &["--time-col", "-1"][..],
            "invalid value for --time-col: -1",
        ),
        (
            "drill",
            &["--start", "later"][..],
            "invalid value for --start: later",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_backfold"))
            .args([command, "unused.csv"])
            .args(flags)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert_eq!(
            String::from_utf8(output.stderr).unwrap().trim(),
            format!("error: {message}")
        );
    }
}
