from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle
from build import make_capsule, recipe
from fleet import apply, blob_id, git, run_recipe
from patchwire import encode, decode, lines


class CodecTests(unittest.TestCase):
    def test_edits(self):
        cases = [
            (b"abc", b""),
            (b"", b"x"),
            (b"xx\nx\n", b"xx\nxx\n"),
            (b"old\r\n", b"new\r\n"),
            (b"\xff\x00", b"\xfe\x00"),
        ]
        for before, after in cases:
            self.assertEqual(run_recipe(before, recipe(before, after)), after)

    def test_tree_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-q")
            path = repo / "s.py"
            path.write_bytes(b"before\n")
            git(repo, "add", "s.py")
            root = git(repo, "write-tree").decode().strip()
            path.write_bytes(b"after\n")
            git(repo, "add", "s.py")
            target = git(repo, "write-tree").decode().strip()
            path.write_bytes(b"dirty\n")
            oid = blob_id(b"before\n")
            doc = make_capsule(
                {root: (target, [])}, [recipe(b"before\n", b"after\n")], {oid: 0}, {oid}
            )
            self.assertEqual(apply(doc, repo, root), target)
            self.assertEqual(git(repo, "write-tree").decode().strip(), target)
            self.assertEqual(path.read_bytes(), b"dirty\n")
            with self.assertRaises(ValueError):
                apply(doc, repo, target)
            doc["recipes"] = [recipe(b"before\n", b"oops\n")]
            with self.assertRaises(ValueError):
                apply(doc, repo, root)

    def test_wire(self):
        source = b"unchanged\0\nold\n--- a/header\n"
        patch = b"diff --git a/p b/p\n--- a/p\n+++ b/p\n@@ -1,3 +1,3 @@\n unchanged\0\n-old\n+new\n --- a/header\n"
        for raw in (b"", patch, patch[:-1], b"arbitrary\0L data"):
            sources = {"p": lines(source)}
            self.assertEqual(decode(encode(raw, sources), sources), raw)

    def test_patch_archive(self):
        rows = [
            b"",
            b" ",
            b"-",
            b"+",
            b"=\x00hello\r\n",
            b" a\n-new\n+old\n+",
            b"bytes\xff",
        ]
        self.assertEqual(bundle.join_columns(bundle.separate(rows)), rows)
        archive, sizes = bundle.encode(rows, [], b"")
        self.assertEqual(bundle.decode(archive, lambda m: b""), rows)
        corrupt = bytearray(archive)
        corrupt[10] ^= 1
        with self.assertRaises(ValueError):
            bundle.decode(bytes(corrupt), lambda m: b"")

    def test_archive_cli_release(self):
        def cli(*args):
            subprocess.run(
                [sys.executable, bundle.__file__, *args],
                check=True,
                capture_output=True,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "clones" / "owner" / "repo"
            repo.mkdir(parents=True)
            git(repo, "init", "-q")
            (repo / "a file").write_bytes(b"one\ntwo\n")
            git(repo, "add", ".")
            git(
                repo,
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "base",
            )
            commit = git(repo, "rev-parse", "HEAD").decode().strip()
            rows = [b"--- a/a file\n--- a/missing\n one\r\n-two\n+three", b"", b" +\n+"]
            for i, row in enumerate(rows):
                (root / f"{i}.patch").write_bytes(row)
            spec = [
                {
                    "repo": "owner/repo",
                    "commit": commit,
                    "patches": [f"{i}.patch" for i in range(3)],
                }
            ]
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps(spec))
            for name, options, clones in (
                ("default", [], root / "clones"),
                ("compact", ["--level", "22"], root / "clones"),
                ("raw", ["--mode", "raw"], root / "absent"),
            ):
                archive = root / (name + ".pfw")
                cli("pack", clones, tasks, archive, *options)
                self.assertEqual(
                    archive.read_bytes()[4],
                    bundle.MODES.index("raw" if name == "raw" else "columns"),
                )
                cli("unpack", clones, archive, root / name)
                restored = [
                    path.read_bytes() for path in sorted((root / name).iterdir())
                ]
                self.assertEqual(restored, rows)


if __name__ == "__main__":
    unittest.main()
