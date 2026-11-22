"""Exercise the installed command from an MCPB, including a real CSV drill-down."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def main():
    archive_path = Path(sys.argv[1])
    with TemporaryDirectory() as directory, ZipFile(archive_path) as archive:
        root = Path(directory)
        manifest = json.loads(archive.read("manifest.json"))
        config = manifest["server"]["mcp_config"]
        platform_key = "win32" if sys.platform == "win32" else sys.platform
        command = {**config, **config["platform_overrides"].get(platform_key, {})}
        binary_name = command["command"].removeprefix("${__dirname}/")
        binary = Path(archive.extract(binary_name, root))
        binary.chmod(0o755)
        fixture = root / "sample.csv"
        fixture.write_text("ts,signal\n0,17\n1,25\n2,42\n3,16\n", encoding="utf-8")
        messages = [
            {"id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "bundle-check", "version": "1"}}},
            {"method": "notifications/initialized"},
            {"id": 2, "method": "tools/list"},
            {"id": 3, "method": "tools/call", "params": {"name": "backfold_drill_csv", "arguments": {"path": str(fixture), "start": 1, "end": 2, "value_col": "signal"}}},
        ]
        payload = "".join(json.dumps({"jsonrpc": "2.0", **message}) + "\n" for message in messages)
        output = subprocess.run(
            [str(binary), *command["args"]], input=payload, text=True,
            capture_output=True, timeout=15, check=True,
        )
        responses = [json.loads(line) for line in output.stdout.splitlines()]
        assert [response["id"] for response in responses] == [1, 2, 3]
        init, listed, call = (response["result"] for response in responses)
        assert init["protocolVersion"] == "2024-11-05"
        assert init["serverInfo"] == {"name": "backfold", "version": manifest["version"]}
        assert {tool["name"] for tool in listed["tools"]} == {tool["name"] for tool in manifest["tools"]}
        assert all(tool["inputSchema"]["type"] == "object" for tool in listed["tools"])
        assert call["isError"] is False
        result = call["content"][0]["text"]
        assert "n**: 2" in result and "25" in result and "42" in result, result
        print(f"{platform_key}: five MCP tools; CSV drill-down returned two rows")


if __name__ == "__main__":
    main()
