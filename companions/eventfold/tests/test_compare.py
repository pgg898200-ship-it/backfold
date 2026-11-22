import io
import unittest
import zlib

from eventfold import pack, split_sse
from eventfold.compare import compare_stream


class CompareTests(unittest.TestCase):
    def test_same_bytes_as_actual_encoders(self):
        for raw in [b"", b"data: hello\n\ndata: hi\n\ndata: hi\n\n", b": comment\r\rfinal"]:
            row = compare_stream(io.BytesIO(raw))
            frames = list(split_sse([raw]))
            self.assertEqual(row["frames"], len(frames))
            self.assertEqual(row["raw_sse_bytes"], len(raw))
            self.assertEqual(row["eventfold_bytes"], len(pack(raw)))
            self.assertEqual(sum(row["eventfold_record_counts"].values()), len(frames))
            gzip = zlib.compressobj(1, zlib.DEFLATED, 28, 3)
            encoded = b"".join(gzip.compress(f) + gzip.flush(zlib.Z_SYNC_FLUSH) for f in frames) + gzip.flush()
            self.assertEqual(row["gzip_level1_window12_mem3_bytes"], len(encoded))
