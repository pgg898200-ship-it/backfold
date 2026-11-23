"""Test the chosen column transform separately for each submission and without the largest record."""

import json
from pathlib import Path
import sys
from urllib.parse import quote
from holdout import bundle, download, digest, SUBMISSIONS, HERE


def report(cache, output):
    cases = json.loads((HERE / "swe.holdout.json").read_text())

    def src(repo, commit, path):
        return download(
            f"https://raw.githubusercontent.com/{repo}/{commit}/{quote(path)}",
            cache / "base" / commit / digest(path.encode()),
            True,
        )

    rows = {}
    for model in SUBMISSIONS:
        raw = (cache / (model + "-all_preds.jsonl")).read_bytes()
        expected = json.loads((HERE / "swe.sources.json").read_text())["archives"][
            model
        ]
        assert digest(raw) == expected
        predictions = {
            r["instance_id"]: (r.get("model_patch") or "").encode()
            for r in map(json.loads, raw.splitlines())
        }
        groups = [(*cases[k], [predictions.get(k, b"")]) for k in sorted(cases)]
        rows[model] = groups
    combined = [
        (r, c, [rows[m][i][2][0] for m in rows])
        for i, (r, c, _) in enumerate(next(iter(rows.values())))
    ]
    biggest = max(range(len(combined)), key=lambda i: sum(map(len, combined[i][2])))
    rows["heldout_without_largest_case"] = combined[:biggest] + combined[biggest + 1 :]
    result = {}
    for label, groups in rows.items():
        manifest, patches, basis = bundle.collect(groups, src)
        raw = bundle.pack_rows(patches)
        header = bundle.zstd(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        )
        wire = bundle.zstd(bundle.separate(patches), basis)
        recovered = bundle.join_columns(bundle.zstd(wire, basis, decode=True))
        assert recovered == patches
        result[label] = {
            "patches": len(patches),
            "raw_zstd19": len(bundle.zstd(raw)),
            "delta_zstd19": 41 + len(header) + len(bundle.zstd(raw, basis)),
            "columns_zstd19": 41 + len(header) + len(wire),
        }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    report(Path(sys.argv[1]), Path(sys.argv[2]))
