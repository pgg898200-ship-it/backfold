"""Export tracked Python sources from upstream Git objects without executing them."""

import argparse
import concurrent.futures
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd)


def git_cat(objects, paths):
    stream = subprocess.run(
        ["git", "-C", str(objects), "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for _, oid in paths),
        capture_output=True,
        check=True,
    ).stdout
    cursor = 0
    for path, oid in paths:
        end = stream.index(b"\n", cursor)
        reported, kind, size = stream[cursor:end].split()
        if reported != oid or kind != b"blob":
            raise ValueError("Unexpected Git object")
        cursor = end + 1
        yield path, stream[cursor : cursor + int(size)]
        cursor += int(size) + 1


def fetch(root, repo, tags, locked):
    name = repo.split("/")[-1]
    objects = root / "objects" / name
    objects.mkdir(parents=True, exist_ok=True)
    git("init", "--bare", "--quiet", cwd=objects)
    entries = {}
    for tag in tags:
        expected = locked.get(f"{repo}@{tag}")
        ref = f"refs/tags/{tag}"
        if subprocess.run(
            ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
            cwd=objects,
            stderr=subprocess.DEVNULL,
        ).returncode:
            git(
                "fetch",
                "--quiet",
                "--depth=1",
                "--no-tags",
                f"https://github.com/{repo}.git",
                f"{ref}:{ref}",
                cwd=objects,
            )
        commit = git("rev-parse", f"{ref}^{{commit}}", cwd=objects).decode().strip()
        if expected and expected != commit:
            raise ValueError(f"Changed tag {repo}@{tag}")
        dest = root / "src" / f"{name}@{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        tree = git("ls-tree", "-rz", commit, cwd=objects)
        paths = []
        for record in tree.rstrip(b"\0").split(b"\0"):
            meta, name_bytes = record.split(b"\t", 1)
            mode, kind, oid = meta.split()
            rel = PurePosixPath(name_bytes.decode())
            if mode in (b"100644", b"100755") and rel.suffix == ".py":
                paths.append((rel, oid))
        batch = git_cat(objects, paths)
        for rel, data in batch:
            path = dest / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        entries[f"{repo}@{tag}"] = commit
        print(f"{repo}@{tag} {commit[:8]}", flush=True)
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    options = parser.parse_args()
    sources = json.loads((HERE / "sources.json").read_text())
    lock_path = HERE / "sources.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    commits = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [
            pool.submit(fetch, options.work.resolve(), repo, tags, lock)
            for repo, tags in sources.items()
        ]
        for job in jobs:
            commits.update(job.result())
    (options.work / "sources.lock.json").write_text(
        json.dumps(commits, sort_keys=True, indent=2) + "\n"
    )
