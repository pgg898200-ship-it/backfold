"""Opt-in EFL1 transport for ASGI streaming responses."""
from __future__ import annotations

from . import DEFAULT_MAX_FRAME_SIZE, MAGIC, _check_limit, _encode, _SSESplitter

_ACCEPT = b"x-accept-eventfold"
_VARY = b"X-Accept-EventFold"
_REPLACED = {
    b"content-type", b"content-length", b"content-encoding", b"etag",
    b"last-modified", b"content-md5", b"digest", b"content-digest",
    b"repr-digest", b"accept-ranges",
}


def _values(headers, name):
    return [v.strip() for k, v in headers if k.lower() == name]


def _tokens(headers, name):
    return {part.strip().lower() for value in _values(headers, name)
            for part in value.split(b",")}


def _eligible(message, headers):
    types = _values(headers, b"content-type")
    return (message["status"] == 200 and len(types) == 1
            and types[0].partition(b";")[0].strip().lower() == b"text/event-stream"
            and not (_tokens(headers, b"content-encoding") - {b"identity"})
            and not _values(headers, b"content-range")
            and not message.get("trailers")
            and b"no-transform" not in _tokens(headers, b"cache-control"))


class EventFoldMiddleware:
    """Encode HTTP 200 SSE for a client that requests X-Accept-EventFold: 1."""

    def __init__(self, app, *, max_frame_size=DEFAULT_MAX_FRAME_SIZE):
        _check_limit(max_frame_size)
        self.app = app
        self.max_frame_size = max_frame_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] == "HEAD":
            return await self.app(scope, receive, send)
        request_headers = scope.get("headers", ())
        opted_in = (_values(request_headers, _ACCEPT) == [b"1"]
                    and b"no-transform" not in _tokens(request_headers, b"cache-control"))
        if opted_in and scope.get("extensions"):
            # Ask the application to stream file bodies via ordinary body messages.
            extensions = {k: v for k, v in scope["extensions"].items()
                          if k not in {"http.response.pathsend", "http.response.zerocopysend"}}
            scope = {**scope, "extensions": extensions}
        splitter = None
        prefix = MAGIC
        previous = b""

        async def send_response(message):
            nonlocal splitter, prefix, previous
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", ()))
                if _eligible(message, headers):
                    if not (_tokens(headers, b"vary") & {_ACCEPT, b"*"}):
                        headers.append((b"vary", _VARY))
                    if opted_in:
                        splitter = _SSESplitter(self.max_frame_size)
                        headers = [(k, v) for k, v in headers if k.lower() not in _REPLACED]
                        headers.append((b"content-type", b"application/x-eventfold"))
                    message = {**message, "headers": headers}
                return await send(message)

            if splitter is None or message["type"] != "http.response.body":
                return await send(message)

            for frame in splitter.feed(message.get("body", b"")):
                record = prefix + _encode(previous, frame)
                previous, prefix = frame, b""
                await send({"type": "http.response.body", "body": record, "more_body": True})

            if not message.get("more_body", False):
                for frame in splitter.finish():
                    prefix += _encode(previous, frame)
                await send({"type": "http.response.body", "body": prefix, "more_body": False})
            elif prefix:
                # Flush headers on the application's first write even for a partial frame.
                await send({"type": "http.response.body", "body": prefix, "more_body": True})
                prefix = b""

        return await self.app(scope, receive, send_response)
