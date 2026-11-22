"""Compare actual application bytes on one SSE connection, with equal flush cadence."""
import zlib
from typing import BinaryIO

from . import DEFAULT_MAX_FRAME_SIZE, MAGIC, _chunks, _encode, split_sse


def compare_stream(source: BinaryIO, *, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> dict:
    """Read one connection. Count complete EFL1 and gzip streams, including trailers."""
    gzip = zlib.compressobj(1, zlib.DEFLATED, 16 + 12, 3)
    gzip6 = zlib.compressobj(6, zlib.DEFLATED, 31, 8)
    ws = zlib.compressobj(6, zlib.DEFLATED, -15, 8)
    gzip6_wire = ws_payload = 0
    previous = b""
    wire, gzip_wire, raw, frames = len(MAGIC), 0, 0, 0
    modes = dict(literal=0, repeat=0, patch=0, two_splice=0)
    for current in split_sse(_chunks(source), max_frame_size=max_frame_size):
        record = _encode(previous, current)
        tag = record[0]
        mode = "literal" if tag < 64 else "patch" if tag < 128 else "repeat" if tag == 128 else "two_splice"
        modes[mode] += 1
        wire += len(record)
        gzip_wire += len(gzip.compress(current)) + len(gzip.flush(zlib.Z_SYNC_FLUSH))
        gzip6_wire += len(gzip6.compress(current)) + len(gzip6.flush(zlib.Z_SYNC_FLUSH))
        # RFC 7692 transport omits the four-byte sync marker at a message boundary.
        ws_payload += len(ws.compress(current)) + len(ws.flush(zlib.Z_SYNC_FLUSH)) - 4
        raw += len(current)
        frames += 1
        previous = current
    gzip_wire += len(gzip.flush(zlib.Z_FINISH))
    gzip6_wire += len(gzip6.flush(zlib.Z_FINISH))
    return {"frames": frames, "raw_sse_bytes": raw, "eventfold_bytes": wire,
            "gzip_level1_window12_mem3_bytes": gzip_wire, "gzip_level6_bytes": gzip6_wire,
            "websocket_deflate6_payload_bytes_excluding_transport": ws_payload,
            "eventfold_record_counts": modes}
