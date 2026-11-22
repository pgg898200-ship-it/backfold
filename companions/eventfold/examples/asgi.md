# Opt in to EFL1 on an ASGI streaming endpoint

Install EventFold in the server's Python environment. For a FastAPI or Starlette app, register the middleware before starting the server.

```python
from eventfold.asgi import EventFoldMiddleware

app.add_middleware(EventFoldMiddleware)
```

For an ASGI application without a middleware registry, export `app = EventFoldMiddleware(app)`. For vLLM's OpenAI-compatible server, add this option to your existing launch command. [vLLM serve options](https://docs.vllm.ai/en/latest/cli/serve/)

```sh
vllm serve "$MODEL" --middleware eventfold.asgi.EventFoldMiddleware
```

On the paired internal client, request the wire format explicitly and check the response media type. This example sends an existing streaming chat payload from `request.json` to a local endpoint and writes the original SSE bytes to stdout.

```python
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from eventfold import unpack_stream

request = Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=Path("request.json").read_bytes(),
    headers={
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Accept-EventFold": "1",
        "Accept-Encoding": "identity",
    },
)
with urlopen(request) as response:
    media_type = response.headers.get_content_type()
    if media_type == "application/x-eventfold":
        unpack_stream(response, sys.stdout.buffer)
    elif media_type == "text/event-stream":
        while chunk := response.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    else:
        raise ValueError(f"Unexpected response type {media_type!r}")
```

Ordinary SSE clients keep their original body chunks. Eligible responses carry `Vary: X-Accept-EventFold`. The middleware transforms HTTP 200 SSE bodies with identity encoding, honors request and response `Cache-Control: no-transform`, and passes through HEAD, range responses, and responses with trailers. It removes representation validators and original lengths from transformed responses. The wire media type is `application/x-eventfold`.

Each response starts with an empty history. Complete frames are sent under the ASGI server's normal backpressure. A trailing CR at the end of a body chunk waits for one more byte or EOF to distinguish CR from CRLF. The default maximum frame size is 16 MiB. Set `max_frame_size` when constructing the middleware to change the limit. Exceeding the limit or disconnecting interrupts the stream. EventFold's earlier complete records may already have been delivered.
