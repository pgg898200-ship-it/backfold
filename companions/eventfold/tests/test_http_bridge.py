from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import threading
import unittest
from urllib.request import Request, urlopen

from eventfold import decode_frames

_example = Path(__file__).resolve().parents[1] / "examples/http_bridge.py"
_spec = importlib.util.spec_from_file_location("http_bridge", _example)
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


@contextmanager
def serving(handler):
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}{bridge.PATH}"
        finally:
            server.shutdown()
            worker.join(timeout=5)


class HTTPBridgeTest(unittest.TestCase):
    def test_first_event_visible_before_upstream_finishes(self):
        first = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\r\n\r\n'
        second = b": keepalive\n\ndata: [DONE]\n\n"
        gate = threading.Event()
        observed = {}
        payload = b'{"model":"local","messages":[],"stream":true}'

        class Origin(BaseHTTPRequestHandler):
            def do_POST(self):
                observed["body"] = self.rfile.read(int(self.headers["Content-Length"]))
                observed["authorization"] = self.headers.get("Authorization")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                self.wfile.write(first)
                self.wfile.flush()
                observed["last_event_released_by_client"] = gate.wait(timeout=4)
                self.wfile.write(second)

            def log_message(self, *_):
                pass

        with serving(Origin) as upstream:
            handler = bridge.make_handler(upstream, "demo-only")
            with serving(handler) as gateway:
                req = Request(gateway, data=payload, headers={"Content-Type":"application/json"})
                with urlopen(req, timeout=5) as response:
                    self.assertEqual(response.headers.get_content_type(), bridge.MEDIA)
                    frames = decode_frames(iter(lambda: response.read1(65536), b""))
                    self.assertEqual(next(frames), first)
                    gate.set()
                    self.assertEqual(b"".join(frames), second)
        self.assertTrue(observed["last_event_released_by_client"])
        self.assertEqual(observed["body"], payload)
        self.assertEqual(observed["authorization"], "Bearer demo-only")
