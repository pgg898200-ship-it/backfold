"""Replay an edit capsule into the Git object database; print the verified result tree."""
import argparse
import base64
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import tempfile

def git(repo, *args, data=None, env=None):
    return subprocess.run(["git", "-C", str(repo), *args], input=data, env=env,
                          check=True, stdout=subprocess.PIPE).stdout

def blob_id(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def load(path):
    doc = json.loads(lzma.decompress(Path(path).read_bytes()))
    if doc.get("format") != 2:
        raise ValueError("Unknown capsule format")
    return doc

def unpack(data):
    return base64.b64decode(data, validate=True)

def run_recipe(source, recipe):
    kind, operations = recipe
    if kind == "replace":
        for old, new in operations:
            source = source.replace(unpack(old), unpack(new))
        return source
    if kind != "splice":
        raise ValueError("Unknown edit instruction")
    parts = []
    position = 0
    for start, length, addition in operations:
        if not position <= start <= start + length <= len(source):
            raise ValueError("Edit outside source")
        parts.extend((source[position:start], unpack(addition)))
        position = start + length
    return b"".join([*parts, source[position:]])

def fingerprint(root, target):
    return hashlib.sha256(bytes.fromhex(root+target)).hexdigest()

def apply(doc, repo, revision, *, forecast=False):
    root = git(repo, "rev-parse", "--verify", "--end-of-options", revision + "^{tree}").decode().strip()
    certificate = doc["roots"].get(root[:doc["root_width"]])
    if certificate is None and not forecast:
        raise ValueError(f"Unrecognized source tree {root}")
    recipes = doc["blobs"]
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(tmp)/"index"))
        git(repo, "read-tree", root, env=env)
        index = git(repo, "ls-files", "--stage", "-z", env=env)
        files = []
        for record in filter(None,index.rstrip(b"\0").split(b"\0")):
            meta, name = record.split(b"\t", 1)
            mode, oid, stage = meta.split()
            key = oid[:doc["blob_width"]].decode()
            if key in recipes and mode in (b"100644", b"100755") and name.endswith(b".py"):
                files.append((name, mode, oid))
        batch = git(repo, "cat-file", "--batch", data=b"".join(oid+b"\n" for _,_,oid in files))
        cursor, records = 0, []
        for name, mode, oid in files:
            end = batch.index(b"\n", cursor)
            reported, kind, length = batch[cursor:end].split()
            if reported != oid or kind != b"blob":
                raise ValueError("Unexpected Git object")
            cursor = end + 1
            source = batch[cursor:cursor+int(length)]
            cursor += int(length) + 1
            changed = run_recipe(source, doc["recipes"][recipes[oid[:doc["blob_width"]].decode()]])
            new_oid = git(repo, "hash-object", "-w", "--stdin", data=changed).strip()
            records.append(mode + b" " + new_oid + b"\t" + name + b"\0")
        git(repo, "update-index", "-z", "--index-info", data=b"".join(records), env=env)
        result = git(repo, "write-tree", env=env).decode().strip()
        if not forecast and fingerprint(root,result) != certificate:
            raise ValueError("Input and output Git trees failed certificate verification")
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("repo", type=Path)
    parser.add_argument("revision", nargs="?", default="HEAD")
    args = parser.parse_args()
    print(apply(load(args.capsule), args.repo, args.revision))
