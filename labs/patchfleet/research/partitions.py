"""Compare stock source-aware zstd on issue- and repository-sized groups."""

import collections
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import quote
from holdout import bundle, download, digest, HERE, SUBMISSIONS


def run(cache, out):
    cases = json.loads((HERE / "swe.holdout.json").read_text())
    pins = json.loads((HERE / "swe.sources.json").read_text())["archives"]
    tracks = []
    for model in SUBMISSIONS:
        data = (cache / (model + "-all_preds.jsonl")).read_bytes()
        assert digest(data) == pins[model]
        tracks.append(
            {
                r["instance_id"]: (r.get("model_patch") or "").encode()
                for r in map(json.loads, data.splitlines())
            }
        )

    def read(repo, base, p):
        return download(
            f"https://raw.githubusercontent.com/{repo}/{base}/{quote(p)}",
            cache / "base" / base / digest(p.encode()),
            True,
        )

    def prepare(key):
        repo, base = cases[key]
        m, records, source = bundle.collect(
            [(repo, base, [t.get(key, b"") for t in tracks])], read
        )
        return repo, records, source

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        data = list(ex.map(prepare, sorted(cases)))

    def measure(group):
        raw = bundle.pack_rows([r for _, rs, _ in group for r in rs])
        basis = b"".join(b for _, _, b in group)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "basis"
            path.write_bytes(basis)
            flags = [f"--patch-from={path}"] if basis else []
            wire = subprocess.check_output(
                [
                    "zstd",
                    "-q",
                    "--ultra",
                    "-22",
                    "--long=27",
                    *flags,
                    f"--stream-size={len(raw)}",
                    "-c",
                ],
                input=raw,
            )
            assert (
                subprocess.check_output(
                    ["zstd", "-q", "-d", "--long=27", *flags, "-c"], input=wire
                )
                == raw
            )
            return len(wire)

    by_repo = collections.defaultdict(list)
    for row in data:
        by_repo[row[0]].append(row)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        per_repo = sum(ex.map(measure, by_repo.values()))
        per_issue = sum(ex.map(measure, ([r] for r in data)))
    out.write_text(
        json.dumps(
            dict(
                repo_count=len(by_repo),
                case_count=len(data),
                zstd22_long_source_by_repo=per_repo,
                zstd22_long_source_by_issue=per_issue,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
