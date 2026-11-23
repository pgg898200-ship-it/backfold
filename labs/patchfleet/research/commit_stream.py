"""Verify every output against real Ruff run with a persistent worktree and native cache."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build import FLAGS
from fleet import git
from history_compile import Compiler

HEAD = "a160f384523ac70173d069954b7084bee79fb68e"
COUNT = 128


def run(repo, target):
    revs = (
        git(repo, "rev-list", "--first-parent", f"--max-count={COUNT}", HEAD)
        .decode()
        .splitlines()[::-1]
    )
    if len(revs) != COUNT or revs[0] != "8bebbbcda51a3574840c0d0cab2aa82535329bd7":
        raise ValueError("Missing pinned first-parent history")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        checkout = root / "src"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        objects = (
            git(repo, "rev-parse", "--path-format=absolute", "--git-path", "objects")
            .decode()
            .strip()
        )
        (checkout / ".git/objects/info/alternates").write_text(objects + "\n")
        with contextlib.closing(Compiler(repo, root / "cache.sqlite3")) as compiler:
            for rev in revs:
                t = time.perf_counter()
                metrics = compiler.compile(rev)
                t1 = time.perf_counter() - t
                t = time.perf_counter()
                git(checkout, "reset", "--quiet", "--hard", rev)
                files = [
                    os.fsdecode(row.split(b"\t", 1)[1])
                    for row in git(checkout, "ls-files", "--stage", "-z").split(b"\0")
                    if row
                    and row[:6] in (b"100644", b"100755")
                    and row.endswith(b".py")
                ]
                flags = [x for x in FLAGS if x != "--no-cache"]
                for i in range(0, len(files), 500):
                    subprocess.run(
                        [
                            "ruff",
                            *flags,
                            "--cache-dir",
                            str(root / "ruff-cache"),
                            "--",
                            *files[i : i + 500],
                        ],
                        cwd=checkout,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                git(checkout, "add", "--update")
                expected = git(checkout, "write-tree").decode().strip()
                t2 = time.perf_counter() - t
                if expected != metrics["tree"]:
                    raise ValueError(
                        f"Ruff mismatch at {rev}: {expected} != {metrics['tree']}"
                    )
                rows.append(
                    dict(
                        commit=rev,
                        python_files=len(files),
                        **metrics,
                        compiler_s=t1,
                        baseline_s=t2,
                    )
                )
        with contextlib.closing(Compiler(repo, root / "cache.sqlite3")) as compiler:
            t = time.perf_counter()
            for row in rows[::-1]:
                assert compiler.compile(row["commit"])["tree"] == row["tree"]
            reverse = time.perf_counter() - t
    result = dict(
        project="scikit-image/scikit-image",
        first=revs[0],
        last=revs[-1],
        commits=len(rows),
        tree_sequence_sha256=hashlib.sha256(
            "".join(r["tree"] for r in rows).encode()
        ).hexdigest(),
        parsed=sum(r["python_parsed"] for r in rows),
        file_instances=sum(r["python_files"] for r in rows),
        opened_subtrees=sum(r["directories_opened"] for r in rows),
        reused_subtrees=sum(r["cached_directories"] for r in rows),
        distinct_output_trees=len({r["tree"] for r in rows}),
        changed_snapshot_count=sum(
            r["tree"]
            != git(repo, "rev-parse", r["commit"] + "^{tree}").decode().strip()
            for r in rows
        ),
        compiler_seconds=sum(r["compiler_s"] for r in rows),
        baseline_seconds=sum(r["baseline_s"] for r in rows),
        compiler_p50_ms=1000 * statistics.median(r["compiler_s"] for r in rows),
        baseline_p50_ms=1000 * statistics.median(r["baseline_s"] for r in rows),
        reverse_replay_seconds=reverse,
    )
    target.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
