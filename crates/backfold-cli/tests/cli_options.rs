use std::process::Command;

#[test]
fn csv_commands_report_invalid_options_before_opening_the_file() {
    for (command, flag, bad_value) in [
        ("digest", "--time-col", "-1"),
        ("digest", "--value-col", "value"),
        ("digest", "--top-k", "2.5"),
        ("digest", "--threshold", "high"),
        ("drill", "--time-col", "-1"),
        ("drill", "--value-col", "value"),
        ("drill", "--start", "yesterday"),
        ("drill", "--end", "tomorrow"),
    ] {
        for value in [None, Some(bad_value)] {
            let result = Command::new(env!("CARGO_BIN_EXE_backfold"))
                .args([command, "unopened.csv", flag])
                .args(value)
                .output()
                .unwrap();
            assert_eq!(result.status.code(), Some(2), "{command} {flag} {value:?}");
            let error = String::from_utf8(result.stderr).unwrap();
            assert!(error.contains(flag), "{error}");
            if let Some(value) = value {
                assert!(error.contains(value), "{error}");
            }
        }
    }
}
