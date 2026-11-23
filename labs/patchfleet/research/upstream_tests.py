"""Run four upstream NetworkX modules against original and reconstructed Git trees."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

MODULES = [
    "networkx/algorithms/tests/test_distance_measures.py",
    "networkx/drawing/tests/test_layout.py",
    "networkx/readwrite/tests/test_gexf.py",
    "networkx/readwrite/tests/test_graphml.py",
]


def test(repo, tree, modules):
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        archive = root / "tree.tar"
        subprocess.run(
            ["git", "-C", str(repo), "archive", "-o", str(archive), tree], check=True
        )
        with tarfile.open(archive) as f:
            f.extractall(root, filter="data")
        report = root / "tests.xml"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--disable-warnings",
                "--junitxml",
                str(report),
                *modules,
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode not in (0, 1) or not report.exists():
            raise RuntimeError(proc.stdout.decode()[-3000:])
        result = {}
        for case in ET.parse(report).iter("testcase"):
            key = case.attrib["classname"] + "::" + case.attrib["name"]
            result[key] = next(
                (x.tag for x in case if x.tag in ("failure", "error", "skipped")),
                "passed",
            )
        if not result:
            raise ValueError("Upstream test collection/execution error")
        return result


if __name__ == "__main__":
    from collections import Counter

    repo = Path(sys.argv[1]).resolve()
    summary = json.loads(
        (Path(__file__).resolve().parents[1] / "artifacts/results.json").read_text()
    )
    row = next(
        x
        for x in summary["releases"]
        if x["release"] == "networkx/networkx@networkx-2.8.8"
    )
    modules = ["networkx"] if "--full" in sys.argv[3:] else MODULES
    before, after = [
        test(repo, row[key], modules) for key in ("source_tree", "target_tree")
    ]
    if before.keys() != after.keys():
        raise ValueError("Test collection changed")
    Path(sys.argv[2]).write_text(
        json.dumps(
            dict(
                release=row["release"],
                modules=modules,
                before=dict(Counter(before.values())),
                after=dict(Counter(after.values())),
                transitions=dict(
                    Counter(before[k] + " -> " + after[k] for k in before)
                ),
                remaining_failures=[k for k, v in after.items() if v == "failure"],
            ),
            indent=2,
        )
        + "\n"
    )
