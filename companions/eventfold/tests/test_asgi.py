import asyncio
import unittest

from eventfold import FrameTooLarge, MAGIC, decode_frames, pack
from eventfold.asgi import EventFoldMiddleware


OPT = [(b"x-accept-eventfold", b"1")]
SSE = [(b"content-type", b"text/event-stream; charset=utf-8")]


def start(headers=SSE, **options):
    return {"type": "http.response.start", "status": 200, "headers": headers, **options}


def body(data=b"", more=False):
    return {"type": "http.response.body", "body": data, "more_body": more}


async def run(messages, *, headers=OPT, method="GET", limit=10000):
    result = []

    async def app(scope, receive, send):
        for message in messages:
            await send(message)

    async def capture(message):
        result.append(message)

    await EventFoldMiddleware(app, max_frame_size=limit)(
        {"type": "http", "method": method, "headers": headers}, None, capture)
    return result


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_reassemble_and_invalidate_headers(self):
        headers = SSE + [(b"content-length", b"3"), (b"etag", b'"sse"'),
                         (b"content-digest", b"abc"), (b"vary", b"Origin"),
                         (b"content-encoding", b"identity"), (b"x-extra", b"1")]
        data = b"data: \xff\r\n\r\n: comment\r\rdata: 2\n\nunfinished"
        messages = [start(headers)] + [body(bytes([byte]), True) for byte in data] + [body()]
        result = await run(messages)
        self.assertIn((b"content-type", b"application/x-eventfold"), result[0]["headers"])
        self.assertIn((b"vary", b"Origin"), result[0]["headers"])
        self.assertIn((b"vary", b"X-Accept-EventFold"), result[0]["headers"])
        self.assertIn((b"x-extra", b"1"), result[0]["headers"])
        for key in (b"content-encoding", b"content-length", b"etag", b"content-digest"):
            self.assertNotIn(key, dict(result[0]["headers"]))
        self.assertEqual(b"".join(m["body"] for m in result[1:]), pack(data))
        self.assertTrue(all(m["more_body"] for m in result[1:-1]))
        self.assertFalse(result[-1]["more_body"])
        self.assertEqual(headers[0], SSE[0])

    async def test_raw_optout_preserves_body_messages_and_existing_vary(self):
        original = [start(SSE + [(b"vary", b"Origin, x-Accept-Eventfold")]),
                    body(b"x\n", True), body(b"\n")]
        for headers in ([], OPT + OPT, [(b"x-accept-eventfold", b"0")],
                        OPT + [(b"cache-control", b"public, No-Transform")]):
            self.assertEqual(await run(original, headers=headers), original)
        raw = await run([start(), body(b"raw")], headers=[])
        self.assertEqual(raw[0]["headers"], SSE + [(b"vary", b"X-Accept-EventFold")])
        self.assertEqual(raw[1], body(b"raw"))

    async def test_bypasses(self):
        for head in (start(status=201), start(trailers=True), start([(b"content-type", b"text/plain")]),
                     start(SSE + [(b"cache-control", b"private, NO-TRANSFORM")]),
                     start(SSE + [(b"content-encoding", b"gzip")]),
                     start(SSE + [(b"content-range", b"bytes 0-0/2")])):
            messages = [head, body(b"z")]
            self.assertEqual(await run(messages), messages)
        self.assertEqual(await run([start(), body()], method="HEAD"), [start(), body()])

    async def test_empty_limit_and_clean_completion(self):
        self.assertEqual((await run([start(), body()]))[-1], body(MAGIC))
        self.assertEqual((await run([start(), body(b"abc")], limit=3))[-1], body(pack(b"abc")))
        with self.assertRaises(FrameTooLarge):
            await run([start(), body(b"aaa", True), body(b"x", True)], limit=3)

    async def test_early_delivery_and_cancellation(self):
        first_received = asyncio.Event()
        release = asyncio.Event()
        ended = asyncio.Event()
        received = []

        async def app(scope, receive, send):
            try:
                await send(start())
                await send(body(b"data: first\r", True))
                await send(body(b"\n\r", True))
                await send(body(b"\n", True))
                await release.wait()
                await send(body(b"data: last\n\n"))
            finally:
                ended.set()

        async def send(message):
            received.append(message)
            if (message["type"] == "http.response.body" and message["body"] != MAGIC):
                first_received.set()

        app = EventFoldMiddleware(app)
        scope = {"type": "http", "method": "GET", "headers": OPT}
        for cancel in (True, False):
            first_received.clear()
            ended.clear()
            received.clear()
            task = asyncio.create_task(app(scope, None, send))
            await asyncio.wait_for(first_received.wait(), 2)
            records = [m["body"] for m in received[1:]]
            self.assertEqual(list(decode_frames(records)), [b"data: first\r\n\r\n"])
            self.assertTrue(all(m["more_body"] for m in received[1:]))
            if cancel:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                release.set()
                await asyncio.wait_for(task, 2)
                self.assertFalse(received[-1]["more_body"])
            self.assertTrue(ended.is_set())
