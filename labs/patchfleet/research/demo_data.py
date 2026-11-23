"""Publish small release witnesses, sourced from the verified migration and Git trees."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fleet import git, load


def make(work):
    home = Path(__file__).resolve().parents[1]
    doc = load(home / "artifacts/numpy2.fleet.xz")
    results = {
        r["release"]: r
        for r in json.loads((home / "artifacts/results.json").read_text())["releases"]
    }
    refs = json.loads((home / "sources.lock.json").read_text())
    cases = [
        (
            "scikit-image/scikit-image@v0.22.0",
            "doc/examples/features_detection/plot_windowed_histogram.py",
        ),
        ("networkx/networkx@networkx-2.8.8", "networkx/readwrite/graphml.py"),
        ("nipy/nibabel@5.2.1", "nibabel/ecat.py"),
    ]
    rows = []
    for key, path in cases:
        owner, tag = key.split("@")
        repo = work / "objects" / owner.split("/")[-1]
        before = git(repo, "rev-parse", refs[key] + ":" + path).decode().strip()
        after = (
            git(repo, "rev-parse", results[key]["target_tree"] + ":" + path)
            .decode()
            .strip()
        )
        rows.append(
            dict(
                release=key,
                path=path,
                source=f"https://raw.githubusercontent.com/{owner}/{refs[key]}/{path}",
                before=before,
                after=after,
                recipe=doc["recipes"][doc["blobs"][before[: doc["blob_width"]]]],
            )
        )
    (home / "site/cases.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    make(Path(sys.argv[1]))
