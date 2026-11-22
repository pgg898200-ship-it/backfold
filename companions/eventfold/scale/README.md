# Thousands of active encoders

Run `python3 scale/run.py` on a machine with C++17 and zlib headers. Each run starts a fresh process, then sends one event to each independent logical stream before advancing to the next round. Each synthetic OpenAI-shaped stream has its own ID and token sequence. The default is 2,048 streams, 96 events each and five runs. Every event is flushed. Gzip uses level 1, a 4 KiB window and memory level 3. Byte counts include stream headers and trailers.

The thread CPU timer (POSIX) covers encoding, updating state and writing encoded bytes to one reusable scratch buffer. It excludes SSE generation, socket I/O, TLS, decoding and allocation of gzip contexts during startup. Each run reports its median round time per event after two warm-up rounds. The summary takes the median of those run medians. This is a throughput cost, not per-event p99 latency.

State counts include each encoder's descriptor, retained vector capacity for the previous event, or bytes requested by zlib's allocator through warm-up. They exclude allocator bookkeeping, inputs, the common output buffer and transport buffers. Sizes apply only to this workload. The header itself imposes no fixed previous-frame size.
