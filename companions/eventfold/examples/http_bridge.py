"""Run a local EFL1 gateway and a client for an OpenAI-compatible SSE endpoint.

  python examples/http_bridge.py serve --upstream http://127.0.0.1:8000/v1/chat/completions
  echo '{"model":"local","stream":true,"messages":[{"role":"user","content":"Hi"}]}' | \
    python examples/http_bridge.py client

The second command writes the original SSE bytes to stdout. Optionally set
EVENTFOLD_UPSTREAM_KEY in the server environment. Each request gets independent
codec history. The gateway binds to loopback; this example's wire leg uses HTTP.
"""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from eventfold import pack_stream, unpack_stream

PATH = "/v1/chat/completions"
MEDIA = "application/x-eventfold"
MAX_REQUEST = 1 << 20


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChunkWriter:
    """Frame each write as an HTTP chunk. Send the final chunk on success."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(f"{len(data):x}\r\n".encode("ascii"))
        self.stream.write(data)
        self.stream.write(b"\r\n")
        return len(data)

    def flush(self):
        self.stream.flush()

    def finish(self):
        self.stream.write(b"0\r\n\r\n")
        self.flush()


def make_handler(upstream: str, key: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.close_connection = True
            if self.path != PATH:
                return self.send_error(HTTPStatus.NOT_FOUND)
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0:
                return self.send_error(HTTPStatus.LENGTH_REQUIRED)
            if length > MAX_REQUEST:
                return self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            body = self.rfile.read(length)
            if len(body) != length:
                return self.send_error(HTTPStatus.BAD_REQUEST, "Incomplete request body")
            headers = {"Accept": "text/event-stream", "Content-Type": "application/json",
                       "Accept-Encoding": "identity"}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            req = Request(upstream, data=body, headers=headers, method="POST")
            try:
                response = build_opener(NoRedirect).open(req, timeout=60)
            except HTTPError as error:
                error.close()
                return self.send_error(HTTPStatus.BAD_GATEWAY, f"Upstream HTTP {error.code}")
            except (OSError, URLError):
                return self.send_error(HTTPStatus.BAD_GATEWAY, "Upstream connection failed")
            with response:
                if response.headers.get_content_type() != "text/event-stream":
                    return self.send_error(HTTPStatus.BAD_GATEWAY, "Expected text/event-stream")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    return self.send_error(HTTPStatus.BAD_GATEWAY, "Expected uncompressed SSE")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", MEDIA)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                writer = ChunkWriter(self.wfile)
                try:
                    pack_stream(response, writer)
                    # HTTPResponse.read1 may return EOF with a nonzero declared
                    # length remaining. Keep that transport failure visible.
                    if response.length not in (None, 0):
                        raise OSError("incomplete upstream HTTP body")
                    writer.finish()
                except (OSError, ValueError, HTTPException):
                    # Closing without the terminating HTTP chunk signals a
                    # partial transfer to the downstream reader.
                    self.log_error("stream interrupted")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Paired HTTP transport for OpenAI-compatible SSE")
    commands = parser.add_subparsers(dest="action", required=True)
    serve = commands.add_parser("serve", help="compress responses from a fixed upstream")
    serve.add_argument("--upstream", required=True, help="full chat completions URL")
    serve.add_argument("--port", type=int, default=8765)
    client = commands.add_parser("client", help="POST stdin JSON and stream decoded SSE to stdout")
    client.add_argument("--gateway", default="http://127.0.0.1:8765" + PATH)
    args = parser.parse_args()
    if args.action == "serve":
        if not args.upstream.startswith(("http://", "https://")):
            parser.error("--upstream must be an HTTP(S) URL")
        handler = make_handler(args.upstream, os.environ.get("EVENTFOLD_UPSTREAM_KEY"))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Listening at http://127.0.0.1:{args.port}{PATH}", file=sys.stderr)
            server.serve_forever()
    else:
        body = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(body) > MAX_REQUEST:
            parser.error("JSON request exceeds the 1 MiB limit")
        req = Request(args.gateway, data=body, headers={"Content-Type": "application/json",
                                                       "Accept": MEDIA}, method="POST")
        with build_opener(NoRedirect).open(req, timeout=60) as response:
            if response.headers.get_content_type() != MEDIA:
                parser.error("gateway did not return EventFold content")
            unpack_stream(response, sys.stdout.buffer)


if __name__ == "__main__":
    main()
