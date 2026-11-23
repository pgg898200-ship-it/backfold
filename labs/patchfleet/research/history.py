"""Compare the pinned codemod to independently authored historical migrations."""

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess

CASES = [
    (
        "scikit-image/scikit-image",
        "7386",
        "ea6c1c637d8e8c377a5e76bd82a7e9a74f9b7b89",
        "1bd2bb953b75126c58b425ae3477fb11221b80fa",
    ),
    (
        "scikit-image/scikit-image",
        "7118",
        "3aeefa39242e6b19004867b7e79aeab7b8cbd6fd",
        "894974a6a471982f0c908f2568c382e6364eaa16",
    ),
    (
        "mne-tools/mne-python",
        "12353",
        "16c17b4dba8381615d57267c01f1f2f310fc522c",
        "ec804a5a8fbbc6034dbd07f18aefb70e3bde0062",
    ),
    (
        "Allen-Tildesley/examples",
        "23",
        "6e16f8e60d6c9799b41a64fdd0cd72cf367f3d91",
        "7aaab92f2d0bf381b640832e8fc11cbd364268e4",
    ),
]


def run(repo, *args, data=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args], input=data, check=True, capture_output=True
    ).stdout


def verify(work, row):
    project, number, base, head = row
    repo = work / (project.split("/")[-1] + "-" + number)
    repo.mkdir(parents=True, exist_ok=True)
    run(repo, "init", "--bare", "--quiet")
    for sha in (base, head):
        if subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", sha], capture_output=True
        ).returncode:
            run(
                repo,
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                "https://github.com/" + project + ".git",
                sha,
            )
    paths = run(
        repo, "diff", "--name-only", "--diff-filter=M", "-z", base, head, "--", "*.py"
    ).split(b"\0")
    groups = {"exact": [], "partial_or_different": [], "unchanged_by_rule": []}
    for name in filter(None, paths):
        path = name.decode()
        old = run(repo, "show", base + ":" + path)
        gold = run(repo, "show", head + ":" + path)
        out = subprocess.run(
            [
                "ruff",
                "check",
                "--isolated",
                "--no-cache",
                "--target-version",
                "py39",
                "--select",
                "NPY201",
                "--fix-only",
                "--stdin-filename",
                path,
                "-",
            ],
            input=old,
            check=True,
            capture_output=True,
        ).stdout
        key = (
            "exact"
            if out == gold
            else "unchanged_by_rule"
            if out == old
            else "partial_or_different"
        )
        groups[key].append(path)
    return dict(project=project, pull=number, base=base, head=head, **groups)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: verify(args.work, row), CASES))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
