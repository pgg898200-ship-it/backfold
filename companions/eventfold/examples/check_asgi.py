"""Exercise interleaved EventFold connections over real loopback HTTP.

Requires FastAPI, Uvicorn and HTTPX. Install EventFold first, then run:
    python examples/check_asgi.py --streams 128
"""
import argparse
import asyncio
import socket

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

from eventfold import pack, unpack
from eventfold.asgi import EventFoldMiddleware


async def probe(count: int):
    app = FastAPI()
    app.add_middleware(EventFoldMiddleware)
    release = asyncio.Event()
    all_received = asyncio.Event()
    first_arrivals = set()

    def raw_event(stream_id: int, part: str) -> bytes:
        return f"data: stream-{stream_id}-{part}\n\n".encode()

    @app.get("/events/{stream_id}")
    async def events(stream_id: int):
        async def generate():
            yield raw_event(stream_id, "first")
            await release.wait()
            yield raw_event(stream_id, "second")
        return StreamingResponse(generate(), media_type="text/event-stream")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(count)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        serve = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            while not server.started:
                if serve.done(): await serve
                await asyncio.sleep(0.01)
            limits = httpx.Limits(max_connections=count, max_keepalive_connections=0)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", limits=limits,
                                         timeout=25, trust_env=False) as http:
                async def visit(i):
                    expected = raw_event(i, "first")
                    first_wire = pack(expected)
                    async with http.stream("GET", f"/events/{i}",
                                           headers={"X-Accept-EventFold": "1"}) as response:
                        response.raise_for_status()
                        assert response.headers["content-type"] == "application/x-eventfold"
                        wire = bytearray()
                        async for chunk in response.aiter_raw():
                            wire.extend(chunk)
                            if i not in first_arrivals and len(wire) >= len(first_wire):
                                assert bytes(wire) == first_wire
                                first_arrivals.add(i)
                                if len(first_arrivals) == count:
                                    all_received.set()
                        assert unpack(wire) == expected + raw_event(i, "second")
                clients = asyncio.gather(*(visit(i) for i in range(count)))
                try:
                    # Every client must receive a complete event before *any* producer finishes.
                    await asyncio.wait_for(all_received.wait(), 20)
                    release.set()
                    await clients
                finally:
                    release.set()
                    if not clients.done(): clients.cancel()
                    await asyncio.gather(clients, return_exceptions=True)
                raw = await http.get("/events/0")
                assert raw.headers["content-type"].startswith("text/event-stream")
                assert raw.content == raw_event(0, "first") + raw_event(0, "second")
        finally:
            release.set()
            server.should_exit = True
            await serve
    print(f"{count} independent HTTP streams delivered their first event before any finished; "
          f"all {count * 2} encoded events and the ordinary SSE response matched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.streams <= 1024:
        parser.error("--streams must be in 1..1024")
    asyncio.run(probe(args.streams))
