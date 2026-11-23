"""Stock Git plus Ruff on changed paths and previously fixed paths."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from commit_stream import HEAD, COUNT
from build import FLAGS
from fleet import git


def run(repo, target, reference):
    expected = json.loads(reference.read_text())
    revs = (
        git(repo, "rev-list", "--first-parent", f"--max-count={COUNT}", HEAD)
        .decode()
        .splitlines()[::-1]
    )
    roots, elapsed = [], []
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        repo_copy = Path(tmp) / "src"
        repo_copy.mkdir()
        git(repo_copy, "init", "--quiet")
        objects = (
            git(repo, "rev-parse", "--path-format=absolute", "--git-path", "objects")
            .decode()
            .strip()
        )
        (repo_copy / ".git/objects/info/alternates").write_text(objects + "\n")
        prior_edits = set()
        last = None
        for rev in revs:
            t = time.perf_counter()
            git(repo_copy, "reset", "--hard", "--quiet", rev)
            if last is None:
                names = git(repo, "ls-tree", "-r", "-z", "--name-only", rev)
            else:
                names = git(
                    repo, "diff", "--no-renames", "--name-only", "-z", last, rev
                )
            candidates = prior_edits | {
                n for n in names.split(b"\0") if n.endswith(b".py")
            }
            paths = [
                os.fsdecode(p)
                for p in sorted(candidates)
                if (repo_copy / os.fsdecode(p)).is_file()
                and not (repo_copy / os.fsdecode(p)).is_symlink()
            ]
            flags = [x for x in FLAGS if x != "--no-cache"]
            for i in range(0, len(paths), 500):
                subprocess.run(
                    [
                        "ruff",
                        *flags,
                        "--cache-dir",
                        str(Path(tmp) / "ruff"),
                        "--",
                        *paths[i : i + 500],
                    ],
                    cwd=repo_copy,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            git(repo_copy, "add", "--update")
            roots.append(git(repo_copy, "write-tree").decode().strip())
            prior_edits = set(
                filter(
                    None,
                    git(repo_copy, "diff", "--cached", "--name-only", "-z", rev).split(
                        b"\0"
                    ),
                )
            )
            total += len(paths)
            elapsed.append(time.perf_counter() - t)
            last = rev
    digest = hashlib.sha256("".join(roots).encode()).hexdigest()
    if digest != expected["tree_sequence_sha256"]:
        raise ValueError("Reference roots differ")
    target.write_text(
        json.dumps(
            dict(
                commits=len(revs),
                tree_sequence_sha256=digest,
                paths_submitted=total,
                seconds=sum(elapsed),
                p50_ms=statistics.median(elapsed) * 1000,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]))
