# Backfold for MCP clients

Ask your client to summarize a local time-series CSV, then inspect the original rows around a timestamp. Backfold also provides Kalman smoothing, principal component analysis and multi-stream sifting. Supply a local path accessible to the server process. CSV input uses simple comma-separated numeric columns with optional headers.

The bundle contains the Backfold 0.3.0 executables for Apple Silicon macOS, Linux x86-64 and Windows x86-64. Each runs locally over stdio. A client that supports MCP bundles can open the [release bundle](https://github.com/pgg898200-ship-it/backfold/releases/download/v0.3.0/backfold-0.3.0.mcpb). In a client with manual stdio configuration, use the executable for your platform from the [release](https://github.com/pgg898200-ship-it/backfold/releases/tag/v0.3.0) as the command and `["mcp"]` as its arguments.

For a first call, save the following as `sample.csv`.

```csv
ts,signal
0,17
1,25
2,42
3,16
```

Ask your client to call `backfold_drill_csv` with the absolute CSV path, `value_col="signal"`, `start=1` and `end=2`. The result should contain the two observations 25 and 42.

## Reproduce the bundle

Run these commands from the repository root with Python 3.9 or later. The builder verifies the SHA-256 of each downloaded executable. The smoke test launches the current platform's bundled executable, initializes MCP, checks the tool list and calls the CSV drill-down.

```sh
python integrations/mcpb/build.py
python integrations/mcpb/smoke.py backfold-0.3.0.mcpb
```

The [registry descriptor](server.json) records the bundle digest and versioned download URL.
