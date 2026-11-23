import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build import make_capsule,recipe
from fleet import apply,blob_id,git,run_recipe
from patchwire import encode,decode,lines

class CodecTests(unittest.TestCase):
    def test_edits(self):
        cases=[(b"abc",b""),(b"",b"x"),(b"xx\nx\n",b"xx\nxx\n"),(b"old\r\n",b"new\r\n"),(b"\xff\x00",b"\xfe\x00")]
        for before,after in cases:
            self.assertEqual(run_recipe(before,recipe(before,after)),after)
    def test_tree_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)
            git(repo,"init","-q")
            path=repo/"s.py"
            path.write_bytes(b"before\n")
            git(repo,"add","s.py")
            root=git(repo,"write-tree").decode().strip()
            path.write_bytes(b"after\n")
            git(repo,"add","s.py")
            target=git(repo,"write-tree").decode().strip()
            path.write_bytes(b"dirty\n")
            oid=blob_id(b"before\n")
            doc=make_capsule({root:(target,[])},[recipe(b"before\n",b"after\n")],{oid:0},{oid})
            self.assertEqual(apply(doc,repo,root),target)
            self.assertEqual(git(repo,"write-tree").decode().strip(),target)
            self.assertEqual(path.read_bytes(),b"dirty\n")
            with self.assertRaises(ValueError):
                apply(doc,repo,target)
            doc["recipes"]=[recipe(b"before\n",b"oops\n")]
            with self.assertRaises(ValueError):
                apply(doc,repo,root)
    def test_wire(self):
        source=b"unchanged\0\nold\n--- a/header\n"
        patch=b"diff --git a/p b/p\n--- a/p\n+++ b/p\n@@ -1,3 +1,3 @@\n unchanged\0\n-old\n+new\n --- a/header\n"
        for raw in (b"",patch,patch[:-1],b"arbitrary\0L data"):
            sources={"p":lines(source)}
            self.assertEqual(decode(encode(raw,sources),sources),raw)
if __name__=="__main__":
    unittest.main()
