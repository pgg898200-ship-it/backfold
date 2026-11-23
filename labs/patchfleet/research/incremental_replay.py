import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from incremental import migrate
from build import HERE


def replay(work):
    expected = {
        r["release"]: r["target_tree"]
        for r in json.loads((HERE / "artifacts/results.json").read_text())["releases"]
    }
    tags = json.loads((HERE / "sources.json").read_text())
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, versions in sorted(tags.items()):
            repo = work / "objects" / name.split("/")[-1]
            for i, ver in enumerate(versions):
                row = migrate(repo, ver, Path(tmp) / "cache.sqlite3")
                ref = f"{name}@{ver}"
                if expected[ref] != row["target_tree"]:
                    raise ValueError(f"Tree mismatch at {ref}")
                rows.append(dict(release=ref, last=i == len(versions) - 1, **row))
    return rows


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(
        json.dumps(replay(Path(sys.argv[1]).resolve()), indent=2) + "\n"
    )
