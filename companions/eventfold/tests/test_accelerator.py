"""The published extension and fallback must make the same wire decisions."""
from pathlib import Path
import random
import unittest

from eventfold import _encode, _encode_python, encode_frames, split_sse

try:
    from eventfold import _native
except ImportError:
    _native = None


class AcceleratorTests(unittest.TestCase):
    def test_encoder_selection(self):
        self.assertIs(_encode, _native.encode if _native else _encode_python)

    @unittest.skipUnless(_native, "install the optional C extension to check both encoders")
    def test_fixture_differential(self):
        root = Path(__file__).resolve().parents[1]
        for path in sorted((root / "fixtures").rglob("*.sse")):
            with self.subTest(path=path.name):
                prev = b""
                for frame in split_sse([path.read_bytes()]):
                    self.assertEqual(_native.encode(prev, frame), _encode_python(prev, frame))
                    prev = frame

    @unittest.skipUnless(_native, "install the optional C extension to check both encoders")
    def test_diff_deterministic_mutations(self):
        rng = random.Random(735)
        for size in [0, 1, 63, 129, 255, 1019, 1024, 1025, 8192, 65536]:
            previous = rng.randbytes(size)
            for _ in range(24):
                left, right = sorted([rng.randrange(size + 1), rng.randrange(size + 1)])
                frame = previous[:left] + rng.randbytes(rng.randrange(64)) + previous[left:right]
                frame += rng.randbytes(rng.randrange(64)) + previous[right:]
                self.assertEqual(_native.encode(previous, frame), _encode_python(previous, frame))
                self.assertEqual(_native.encode(frame, frame), b"\x80")

    def test_public_api_accepts_bytes_like_frames(self):
        frames = [memoryview(b"a"), bytearray(b"bbb")]
        self.assertEqual(list(encode_frames(frames)), [b"EFL1", b"\x01a", b"\x03bbb"])
