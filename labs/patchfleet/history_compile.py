"""Incrementally migrate Git snapshots, skipping entire previously compiled directories."""

import argparse
import contextlib
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from build import FLAGS, recipe
from fetch import git_cat
from fleet import git, run_recipe


class Objects:
    def __init__(self, repo, check=False):
        args = [
            "git",
            "-C",
            str(repo),
            "cat-file",
            "--batch-check" if check else "--batch",
        ]
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.check = check

    def read(self, oid):
        self.proc.stdin.write(oid.hex().encode() + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().split()
        if len(header) == 2 and header[1] == b"missing":
            return None
        if len(header) != 3 or header[0] != oid.hex().encode():
            raise ValueError("Unexpected Git batch response")
        if self.check:
            return header[1]
        data = self.proc.stdout.read(int(header[2]))
        if self.proc.stdout.read(1) != b"\n":
            raise ValueError("Truncated Git object")
        return data

    def close(self):
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.wait()


def entries(tree):
    offset = 0
    while offset < len(tree):
        end = tree.index(b"\0", offset)
        mode, name = tree[offset:end].split(b" ", 1)
        yield mode, name, tree[end + 1 : end + 21]
        offset = end + 21


class Compiler:
    def __init__(self, repo, cache):
        self.repo = Path(repo).resolve()
        if git(self.repo, "rev-parse", "--show-object-format").strip() != b"sha1":
            raise ValueError("This version requires SHA-1 Git object format")
        if subprocess.check_output(["ruff", "--version"]).strip() != b"ruff 0.11.13":
            raise ValueError("Install ruff==0.11.13")
        self.db = sqlite3.connect(cache, timeout=60)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS trees_n201_v1(path BLOB,src BLOB,dst BLOB,PRIMARY KEY(path,src))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS files_n201_v1(path BLOB,src BLOB,program TEXT,PRIMARY KEY(path,src))"
        )
        self.objects = Objects(self.repo)
        self.exists = Objects(self.repo, check=True)

    def close(self):
        self.db.close()
        self.objects.close()
        self.exists.close()

    def compile(self, revision):
        root = bytes.fromhex(
            git(
                self.repo,
                "rev-parse",
                "--verify",
                "--end-of-options",
                revision + "^{tree}",
            )
            .decode()
            .strip()
        )
        expanded = {}
        known = {}
        pending = {}
        cached_edits = {}
        skipped = 0

        def visit(prefix, oid):
            nonlocal skipped
            row = self.db.execute(
                "SELECT dst FROM trees_n201_v1 WHERE path=? AND src=?", (prefix, oid)
            ).fetchone()
            if row and (row[0] == oid or self.exists.read(row[0]) == b"tree"):
                known[prefix] = row[0]
                skipped += 1
                return
            source = self.objects.read(oid)
            expanded[prefix] = (oid, list(entries(source)))
            for mode, name, child in expanded[prefix][1]:
                path = prefix + name
                if mode == b"40000":
                    visit(path + b"/", child)
                elif mode in (b"100644", b"100755") and name.endswith(b".py"):
                    hit = self.db.execute(
                        "SELECT program FROM files_n201_v1 WHERE path=? AND src=?",
                        (path, child),
                    ).fetchone()
                    if hit is None:
                        pending[path] = child
                    elif hit[0]:
                        cached_edits[path] = (child, json.loads(hit[0]))

        visit(b"", root)
        changed = {}

        def store(data, kind):
            return bytes.fromhex(
                git(self.repo, "hash-object", "-w", "-t", kind, "--stdin", data=data)
                .decode()
                .strip()
            )

        raw = (
            dict(
                git_cat(
                    self.repo, [(p, oid.hex().encode()) for p, oid in pending.items()]
                )
            )
            if pending
            else {}
        )
        updates = []
        if raw:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                paths = list(raw)
                for p, data in raw.items():
                    dest = base / p.decode()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                for i in range(0, len(paths), 500):
                    subprocess.run(
                        [
                            "ruff",
                            *FLAGS,
                            "--",
                            *(p.decode() for p in paths[i : i + 500]),
                        ],
                        cwd=base,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                for p, data in raw.items():
                    after = (base / p.decode()).read_bytes()
                    program = None
                    if data != after:
                        changed[p] = store(after, "blob")
                        program = json.dumps(recipe(data, after), separators=(",", ":"))
                    updates.append((p, pending[p], program))
        for path, (child, edit) in cached_edits.items():
            changed[path] = store(run_recipe(self.objects.read(child), edit), "blob")
        new_trees = []
        for prefix, (oid, rows) in reversed(list(expanded.items())):
            result = bytearray()
            different = False
            for mode, name, child in rows:
                path = prefix + name
                target = (
                    known[path + b"/"] if mode == b"40000" else changed.get(path, child)
                )
                different |= target != child
                result.extend(mode + b" " + name + b"\0" + target)
            known[prefix] = store(bytes(result), "tree") if different else oid
            new_trees.append((prefix, oid, known[prefix]))
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO files_n201_v1 VALUES (?,?,?)", updates
            )
            self.db.executemany(
                "INSERT OR REPLACE INTO trees_n201_v1 VALUES (?,?,?)", new_trees
            )
        return dict(
            tree=known[b""].hex(),
            directories_opened=len(expanded),
            cached_directories=skipped,
            python_parsed=len(raw),
            bytes_parsed=sum(map(len, raw.values())),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("revisions", nargs="+")
    a = p.parse_args()
    with contextlib.closing(Compiler(a.repo, a.cache)) as compiler:
        for rev in a.revisions:
            print(json.dumps(dict(revision=rev, **compiler.compile(rev))))
