"""Compile Ruff's verified per-release outputs into replayable Git trees."""

import argparse
import base64
import collections
import difflib
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from fleet import apply, blob_id, git, run_recipe

HERE = Path(__file__).resolve().parent
FLAGS = [
    "check",
    "--isolated",
    "--no-cache",
    "--no-respect-gitignore",
    "--target-version",
    "py39",
    "--select",
    "NPY201",
    "--fix-only",
]


def b64(data):
    return base64.b64encode(data).decode()


def compact(value):
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()


def recipe(left, right, *, literal=False):
    a, b = left.splitlines(keepends=True), right.splitlines(keepends=True)
    offsets = [0]
    for line in a:
        offsets.append(offsets[-1] + len(line))
    operations, replacements = [], {}
    word = re.compile(rb"[A-Za-z_][A-Za-z_0-9]*|[^A-Za-z_0-9]+|[0-9]+")
    for op, i, j, k, l in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if op == "equal":
            continue
        old, new = b"".join(a[i:j]), b"".join(b[k:l])
        start, end = 0, 0
        while start < min(len(old), len(new)) and old[start] == new[start]:
            start += 1
        while end < min(len(old), len(new)) - start and old[-1 - end] == new[-1 - end]:
            end += 1
        operations.append(
            [
                offsets[i] + start,
                len(old) - start - end,
                b64(new[start : len(new) - end]),
            ]
        )
        words_old, words_new = word.findall(old), word.findall(new)
        for tag, u, v, w, x in difflib.SequenceMatcher(
            None, words_old, words_new, autojunk=False
        ).get_opcodes():
            if tag == "replace":
                replacements[b"".join(words_old[u:v])] = b"".join(words_new[w:x])
    sub = ["replace", [[b64(x), b64(y)] for x, y in replacements.items()]]
    if not literal and replacements and run_recipe(left, sub) == right:
        return sub
    return ["splice", operations]


def reference_index(repo, rev, fixed):
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(d) / "index"))
        git(repo, "read-tree", rev, env=env)
        records = []
        for path, data in fixed.items():
            entry = git(repo, "ls-files", "-z", "--stage", "--", path, env=env)
            mode = entry.split(b" ", 1)[0]
            oid = git(repo, "hash-object", "-w", "--stdin", data=data).strip()
            records.append(mode + b" " + oid + b"\t" + path.encode() + b"\0")
        git(repo, "update-index", "-z", "--index-info", data=b"".join(records), env=env)
        return git(repo, "write-tree", env=env).decode().strip()


def compare_sizes(raw):
    zstd = subprocess.run(
        ["zstd", "-q", "-19", "--stdout"], input=raw, capture_output=True, check=True
    ).stdout
    return {
        "raw": len(raw),
        "xz9": len(lzma.compress(raw, preset=9)),
        "zstd19": len(zstd),
    }


def prefix_width(selected, all_ids):
    for width in range(1, 41):
        counts = collections.Counter(oid[:width] for oid in all_ids)
        if all(counts[oid[:width]] == 1 for oid in selected):
            return width
    raise ValueError("Ambiguous full object IDs")


def make_capsule(roots, rules, file_rules, seen):
    from fleet import fingerprint

    rw, bw = prefix_width(roots, roots), prefix_width(file_rules, seen)
    return dict(
        format=2,
        root_width=rw,
        blob_width=bw,
        roots={
            key[:rw]: fingerprint(key, target) for key, (target, _) in roots.items()
        },
        blobs={key[:bw]: value for key, value in file_rules.items()},
        recipes=rules,
    )


def build(work, out):
    binary = shutil.which("ruff")
    if (
        not binary
        or subprocess.check_output([binary, "--version"]).decode().strip()
        != "ruff 0.11.13"
    ):
        raise ValueError("Requires ruff==0.11.13")
    sources = json.loads((HERE / "sources.lock.json").read_text())
    roots, rules, ids, stats, timing, original, seen = {}, [], {}, [], {}, {}, set()
    diffs, minimal_diffs, by_repo, offset_roots = (
        [],
        [],
        collections.defaultdict(list),
        {},
    )
    file_rules = {}
    train_rules, train_seen, heldout = {}, set(), set()
    tags = json.loads((HERE / "sources.json").read_text())
    for owner, versions in tags.items():
        heldout.add(owner + "@" + versions[-1])
    start = time.perf_counter()
    for ref, commit in sorted(sources.items()):
        owner, tag = ref.split("@")
        name = owner.split("/")[-1]
        repo = work / "objects" / name
        source_dir = work / "src" / f"{name}@{tag}"
        root = git(repo, "rev-parse", commit + "^{tree}").decode().strip()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "src"
            shutil.copytree(source_dir, dest)
            files = sorted(f.relative_to(dest) for f in dest.rglob("*.py"))
            before = {p: (dest / p).read_bytes() for p in files}
            tracked = git(repo, "ls-tree", "-rz", root)
            tracked_py = {}
            for record in tracked.rstrip(b"\0").split(b"\0"):
                meta, path = record.split(b"\t", 1)
                mode, kind, oid = meta.split()
                if mode in (b"100644", b"100755") and path.endswith(b".py"):
                    tracked_py[Path(os.fsdecode(path))] = oid.decode()
            if set(before) != set(tracked_py) or any(
                blob_id(data) != tracked_py[p] for p, data in before.items()
            ):
                raise ValueError(f"Export differs from tracked Python sources: {ref}")
            t = time.perf_counter()
            for i in range(0, len(files), 500):
                subprocess.run(
                    [
                        binary,
                        *FLAGS,
                        "--",
                        *(str(dest / p) for p in files[i : i + 500]),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            elapsed = time.perf_counter() - t
            fixed, edits, offsets = {}, [], []
            for p, data in before.items():
                out_bytes = (dest / p).read_bytes()
                source_id = tracked_py[p]
                seen.add(source_id)
                if ref not in heldout:
                    train_seen.add(source_id)
                if data != out_bytes:
                    fixed[p.as_posix()] = out_bytes
                    edit = recipe(data, out_bytes)
                    if run_recipe(data, edit) != out_bytes:
                        raise AssertionError(p)
                    key = compact(edit)
                    if key not in ids:
                        ids[key] = len(rules)
                        rules.append(edit)
                    file_rules[source_id] = ids[key]
                    if ref not in heldout:
                        train_rules[source_id] = ids[key]
                    edits.append([p.as_posix(), ids[key]])
                    offsets.append(
                        [p.as_posix(), recipe(data, out_bytes, literal=True)]
                    )
                    if source_id in original and original[source_id] != blob_id(
                        out_bytes
                    ):
                        raise ValueError("Path-sensitive result")
                    original[source_id] = blob_id(out_bytes)
            target = reference_index(repo, commit, fixed)
            roots[root] = [target, edits]
            patch = git(
                repo,
                "diff",
                "--no-ext-diff",
                "--no-renames",
                "--binary",
                root,
                target,
                "--",
            )
            diffs.append(patch)
            short = git(
                repo,
                "diff",
                "--no-ext-diff",
                "--no-renames",
                "--binary",
                "--unified=0",
                "--minimal",
                root,
                target,
                "--",
            )
            short = b"".join(
                re.sub(rb"^(@@ .*? @@)[^\n]*", rb"\1", line)
                for line in short.splitlines(keepends=True)
                if not line.startswith((b"diff --git ", b"index "))
            )
            if short:
                with tempfile.TemporaryDirectory() as td:
                    env = dict(os.environ, GIT_INDEX_FILE=str(Path(td) / "index"))
                    git(repo, "read-tree", root, env=env)
                    git(
                        repo,
                        "apply",
                        "--cached",
                        "--unidiff-zero",
                        "--whitespace=nowarn",
                        "-",
                        data=short,
                        env=env,
                    )
                    assert git(repo, "write-tree", env=env).decode().strip() == target
            minimal_diffs.append(short)
            offset_roots[root] = [target, offsets]
            by_repo[name].append((root, target))
            stats.append(
                dict(
                    release=ref,
                    source_tree=root,
                    target_tree=target,
                    files=len(files),
                    bytes=sum(map(len, before.values())),
                    changed=len(fixed),
                    patch_bytes=len(patch),
                )
            )
            timing[ref] = {"ruff_seconds": elapsed}
    build_seconds = time.perf_counter() - start
    doc = make_capsule(roots, rules, file_rules, seen)
    forecast_doc = make_capsule({}, rules, train_rules, train_seen)
    # Full IDs for unverified transfer prevent aliases from truncated training IDs.
    forecast_doc.update(blobs=train_rules, blob_width=40)
    transfer = []
    capsule = lzma.compress(compact(doc), preset=9)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numpy2.fleet.xz").write_bytes(capsule)
    thin_bytes = 0
    for project, pairs in by_repo.items():
        revisions = "".join(f"{t}\n" for _, t in pairs) + "".join(
            f"^{s}\n" for s, _ in pairs
        )
        pack = git(
            work / "objects" / project,
            "pack-objects",
            "--stdout",
            "--revs",
            "--thin",
            "--no-reuse-delta",
            "--no-reuse-object",
            "--threads=1",
            "--window=250",
            "--depth=50",
            data=revisions.encode(),
        )
        thin_bytes += len(pack)
    for row in stats:
        ref = row["release"]
        repo = work / "objects" / ref.split("@")[0].split("/")[-1]
        t = time.perf_counter()
        assert apply(doc, repo, sources[ref]) == row["target_tree"]
        timing[ref]["replay_seconds"] = time.perf_counter() - t
        if ref in heldout:
            guess = apply(forecast_doc, repo, sources[ref], forecast=True)
            actual = set(
                git(
                    repo,
                    "diff",
                    "--name-only",
                    "-z",
                    row["source_tree"],
                    row["target_tree"],
                ).split(b"\0")
            ) - {b""}
            different = set(
                git(repo, "diff", "--name-only", "-z", guess, row["target_tree"]).split(
                    b"\0"
                )
            ) - {b""}
            transfer.append(
                dict(
                    release=ref,
                    exact_tree=guess == row["target_tree"],
                    expected_edits=len(actual),
                    exact_changed_files=len(actual - different),
                    extra_edits=len(different - actual),
                )
            )
    payload = {
        "leave_latest_release_out": transfer,
        "rule": "Ruff 0.11.13 NPY201 safe fixes, isolated, py39",
        "releases": stats,
        "unique_inputs": len(seen),
        "unique_changed_inputs": len(original),
        "distinct_recipes": len(rules),
        "recipe_types": dict(collections.Counter(r[0] for r in rules)),
        "capsule_bytes": len(capsule),
        "capsule_sha256": hashlib.sha256(capsule).hexdigest(),
        "diffs_concatenated": compare_sizes(b"".join(diffs)),
        "minimal_diffs_concatenated": compare_sizes(b"".join(minimal_diffs)),
        "plain_splice_manifest": compare_sizes(compact(offset_roots)),
        "decoder_xz9_bytes": len(
            lzma.compress((HERE / "fleet.py").read_bytes(), preset=9)
        ),
        "thin_git_packs_bytes": thin_bytes,
        "decoder_source_bytes": (HERE / "fleet.py").stat().st_size,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (work / "timings.json").write_text(
        json.dumps({"build_seconds": build_seconds, "per_release": timing}, indent=2)
        + "\n"
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "releases"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    build(args.work.resolve(), args.out.resolve())
