import hashlib
import json
from pathlib import Path
import unittest

from eventfold import split_sse


class CorpusTests(unittest.TestCase):
    def test_manifests(self):
        corpus = Path(__file__).resolve().parents[1] / "fixtures"
        for directory in [corpus, corpus / "holdout"]:
            entries = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(sorted(r["file"] for r in entries), sorted(p.name for p in directory.glob("*.sse")))
            for row in entries:
                with self.subTest(file=row["file"]):
                    data = (directory / row["file"]).read_bytes()
                    self.assertEqual(len(data), row["sse_bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), row["sse_sha256"])
                    self.assertEqual(len(list(split_sse([data]))), row["frames"])
