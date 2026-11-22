"""Native filter tests at the public pipe boundary."""
import os
from pathlib import Path
import random
import select
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / ("eventfold.exe" if os.name == "nt" else "eventfold")

@unittest.skipUnless(BIN.exists(), "make all to test the native filter")
class NativeStreamTests(unittest.TestCase):
    def run_filter(self, mode, data, *extra):
        return subprocess.run([str(BIN), mode, *extra], input=data, capture_output=True)

    def roundtrip(self, data):
        encoded = self.run_filter("pack", data)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        self.assertTrue(encoded.stdout.startswith(b"EFL1"))
        decoded = self.run_filter("unpack", encoded.stdout)
        self.assertEqual(decoded.returncode, 0, decoded.stderr)
        self.assertEqual(decoded.stdout, data)
        return encoded.stdout

    def test_opaque_inputs_and_eof(self):
        rng = random.Random(122)
        for data in (b"", b"\n", b"\r\r", b"\r\n", b"data: a\r\n\r\n\n:keep\r\rfinal\x00",
                     rng.randbytes(187493), b": " + b"A" * 190000 + b"\n\n" + b": " + b"A" * 190000):
            with self.subTest(length=len(data)):
                self.roundtrip(data)

    def test_fixtures(self):
        for path in sorted((ROOT / "fixtures").rglob("*.sse")):
            with self.subTest(source=path.name):
                self.roundtrip(path.read_bytes())

    def test_bad_records_and_limits(self):
        for invalid in (b"", b"EFL0", b"EFL1\xff", b"EFL1\x02a", b"EFL1\x40\x01\x00"):
            self.assertNotEqual(self.run_filter("unpack", invalid).returncode, 0)
        data = b"abcdefghijklmnopqrst"
        packed = self.roundtrip(data)
        for mode, src in (("pack", data), ("unpack", packed)):
            self.assertNotEqual(self.run_filter(mode, src, "--max-frame", "19").returncode, 0)
            self.assertEqual(self.run_filter(mode, src, "--max-frame", "20").returncode, 0)

    @unittest.skipIf(os.name == "nt", "select on pipes needs POSIX")
    def test_delivery_while_input_is_still_open(self):
        raw = b"data: first\r\n\r\n"
        encoded = self.roundtrip(raw)
        for mode, data, expected in (("pack", raw, encoded), ("unpack", encoded, raw)):
            with self.subTest(mode=mode):
                p = subprocess.Popen([str(BIN), mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                try:
                    for byte in data:
                        p.stdin.write(bytes([byte]))
                        p.stdin.flush()
                    got = b""
                    while len(got) < len(expected):
                        self.assertTrue(select.select([p.stdout], [], [], 3)[0], "output delayed until EOF")
                        part = os.read(p.stdout.fileno(), len(expected)-len(got))
                        self.assertTrue(part)
                        got += part
                    self.assertEqual(got, expected)
                finally:
                    p.stdin.close()
                    p.stdout.close()
                    p.wait(timeout=5)
                self.assertEqual(p.returncode, 0)
