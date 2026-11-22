# EventFold SSE codec

Send each server-sent event across an internal hop as soon as it arrives. EventFold keeps the previous frame, copies unchanged byte ranges, and transmits the edits. The receiving gateway reconstructs the original SSE, including comments, whitespace, field order and line endings. There is no model, tokenizer or shared dictionary.

**Does this help my stream?** Run a captured SSE connection through the local comparison command. It measures EventFold, low-memory gzip, default-window gzip and WebSocket DEFLATE payloads with per-event delivery. No payload leaves your machine. The CLI counts completed gzip streams including trailers; the live benchmark below counts through the final event flush.

```sh
python -m pip install 'git+https://github.com/pgg898200-ship-it/backfold.git@eventfold-v0.1#subdirectory=companions/eventfold'
python -m eventfold compare < capture.sse
```

The C99 hot path is a single header with no allocation or runtime dependencies. The Python implementation and native filter use the same wire format. The browser inspector lets you paste text or open a file locally and inspect every reconstructed frame. [Live browser inspector](https://eventfold.tancurl3.chatgpt.site) · [Wire specification](docs/format.md) · [Protocol and SDK fixtures](fixtures/README.md)

## Use it in an inference gateway

For FastAPI or Starlette, add `app.add_middleware(EventFoldMiddleware)` after importing the class from `eventfold.asgi`. For vLLM, its documented middleware option takes the same import path:

```sh
vllm serve "$MODEL" --middleware eventfold.asgi.EventFoldMiddleware
```

A paired internal client sends `X-Accept-EventFold: 1`; the middleware returns `application/x-eventfold` for eligible SSE responses and declares the request header in `Vary`. Ordinary clients continue receiving SSE. Python uses the optional CPython C encoder when built, with a portable Python fallback and a Python decoder. [Gateway integration and client code](examples/asgi.md)

We exercised the middleware through FastAPI and Uvicorn on 128 simultaneous loopback HTTP connections. Every client decoded its first complete event before any producer was allowed to finish. All 256 events matched, and an ordinary SSE client read the original response. vLLM model serving itself was not exercised. [Reproduce the live HTTP check](examples/check_asgi.py)

## Run the native filter

From this directory, with a C99 compiler and Make installed:

```sh
make
build/eventfold pack < fixtures/openai-text.sse > stream.efl
build/eventfold unpack < stream.efl > restored.sse
cmp fixtures/openai-text.sse restored.sse
```

For a live stream, place the encoder at the inference server and the decoder at the gateway. Both filters read standard input and flush each complete frame. The gateway serves the decoded bytes to ordinary SSE clients. Each process accepts `--max-frame BYTES`; the default is 16 MiB. The source distribution also contains a Python reference gateway and client. [HTTP bridge and usage](examples/http_bridge.py)

```python
from eventfold import decode_frames, encode_frames, split_sse

frames = split_sse([b"data: sun\n\ndata: sea\n\n"])
assert list(decode_frames(encode_frames(frames))) == [b"data: sun\n\n", b"data: sea\n\n"]
```

## Measurements

Across 663 Groq chat-delta events from a public SDK fixture, EventFold encoded 183,368 bytes as **6,290 bytes**, including its stream header. That was 13.2% fewer bytes than context-takeover WebSocket DEFLATE in the same test. The broader result depends on the event mix.

| Native result | Groq chat fixture, 663 events | Ten development streams, 1,101 events | Eight held-out streams, 215 events |
| --- | ---: | ---: | ---: |
| Raw SSE bytes | 183,368 | 323,698 | 45,109 |
| EventFold bytes | **6,290** | 27,353 | 19,717 |
| Fossil best of literal or delta, free selection | 17,975 | 47,228 | 20,857 |
| Linked LZ4 bytes | 13,826 | 33,135 | 12,074 |
| zstd level 3 bytes | 11,188 | 28,499 | 10,198 |
| gzip level 6 bytes | 9,910 | 25,489 | 8,996 |
| WebSocket DEFLATE payload bytes | 7,248 | **20,985** | **8,056** |
| EventFold native encode time per event | 40 ns | 49 ns | 42 ns |

One single-threaded AMD EPYC 9V74 run, seven timing samples per stream. Each codec retained its context and flushed after each event. The live benchmark stops at the final event flush and excludes end-of-stream trailers, WebSocket headers, TLS and network I/O. Timings are amortized thread CPU nanoseconds per event, with suite values weighted by event count.

The separate [Fossil delta replay](fossil/README.md) uses the upstream C implementation and gives the competitor a free per-event choice between raw and delta payloads. EventFold used fewer bytes on 14 of the 18 SDK streams. Fossil was smaller on all three held-out OpenAI Responses streams.

In a separate synthetic run with **2,048 interleaved encoder histories**, EventFold retained **430 KiB** and low-memory gzip retained **51.84 MiB** across the fleet. The measured state ratio was **123×**. The workload used short envelopes and one previous-frame buffer per connection. Application buffers and allocator overhead are excluded. [Raw fleet results](scale/results.json)

See [the native benchmark](bench/README.md) for actual codec versions, per-stream results, raw samples and settings. See [the interleaved-stream experiment](scale/README.md) for operating costs at 2,048 live encoder histories.

## Fit

- **Stable envelopes.** Content deltas often share IDs, model names and field layout with their predecessor. A second edit record can also retain an unchanged middle span when two fields change independently. No JSON parsing is required.
- **Few bytes, many connections.** The native codec borrows the previous frame and caller-owned output storage. A connection does not need a compressor dictionary, and an exact repeat takes one byte.
- **Varied events or long prose.** General-purpose compression can use longer history and compress within a single event. Benchmark your mix, especially Responses lifecycle events, text/tool alternation, and short streams. A gateway that already has effective WebSocket per-message DEFLATE can be a particularly strong alternative.
- **Transport contract.** Both ends of the encoded hop must support EFL1. Give every ordered logical stream its own history and reset it on reconnect. EFL1 assumes a reliable transport with integrity and clean termination semantics. A missing whole final record cannot be detected from EFL1 alone. Keep length-sensitive traffic on a hop with an appropriate padding policy; compressed record lengths differ from upstream lengths.
- **What the figures measure.** Codec CPU time and application payload bytes exclude TCP, TLS, proxy buffering, inter-arrival times and model inference. HTTP-flush behavior also depends on your server and proxies.

## Reproduce and contribute

```sh
make test
node tests/demo.cjs
python3 fossil/run.py
make -C bench
build/bench fixtures 7 2048 > results.csv
python3 bench/summarize.py results.csv
python3 scale/run.py
```

The SDK corpus includes pinned revisions, transformation notes, per-stream digests and upstream licenses. Bring a new provider, a failing wire vector, or aggregate byte counts from a representative capture. For field reports, include the number of independent streams, flush policy and competing compressor settings. Raw production payloads are unnecessary. [Report an EventFold result](https://github.com/pgg898200-ship-it/backfold/issues/new?title=EventFold%20field%20report)

## Related work

io.net documented repeated envelopes on a metered inference-to-gateway hop and published a deployed solution combining token batching with gzip. Our comparison holds event boundaries fixed. Ably offers VCDIFF deltas and Centrifugo supports Fossil message deltas. EventFold targets teams controlling both ends of a streaming connection who want a small embeddable codec they can evaluate locally. [io.net production report](https://io.net/blog/how-we-cut-our-llm-streaming-bandwidth-by-98-with-two-small-changes) · [Ably deltas](https://ably.com/docs/channels/options/deltas) · [Centrifugo delta compression](https://centrifugal.dev/docs/server/delta_compression)

MIT for EventFold. Upstream fixtures and the Fossil benchmark reference retain their own licenses.
