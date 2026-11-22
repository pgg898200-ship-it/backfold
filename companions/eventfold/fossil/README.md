# Fossil delta reference comparison

Run `python3 fossil/run.py` from the repository root on Linux with Python 3 and a C99 compiler. The script verifies the vendored source digests, builds both codecs in a temporary directory, and prints CSV. No Go runtime or network access is required. `results.csv` is the recorded output. There is no timing measurement in this comparison.

Each independent fixture starts with an empty history. For every SSE event we run Fossil SCM's actual `delta_create` on the preceding event and the current event, apply the result using `delta_apply` with checksum verification, and check exact bytes. We report both the resulting delta payload total and the sum of `min(raw_size, delta_size)` over all events. The latter gives Fossil a free choice of raw or delta on each event, plus free message boundaries. It is an optimistic count of the bytes a transport would need. EventFold pays for its four-byte EFL1 header and all record framing. We verify its output using the native CLI.

| Cohort | Fixtures | Events | Original bytes | EventFold | Fossil best of raw/delta | Files smaller with EventFold |
|---|---:|---:|---:|---:|---:|---:|
| Development | 10 | 1,101 | 323,698 | 27,353 | 47,228 | 9 |
| Held out | 8 | 215 | 45,109 | 19,717 | 20,857 | 5 |

On the largest development stream, Groq text, EventFold uses 6,290 bytes and Fossil's optimistic count is 17,975. On all three held-out OpenAI Responses streams, Fossil wins. For example, its tool-use-streaming-1 count is 3,551 bytes versus EventFold's 4,234. Fossil can copy multiple spans at arbitrary offsets in the previous frame. Its standard delta also includes a checksum. EventFold encodes an unchanged prefix, suffix, and optionally one middle span. These figures compare the actual algorithms on these inputs. They do not measure Centrifugo or Ably servers.

## Upstream source

The `upstream/` directory contains the unmodified Fossil SCM C reference used for interoperability tests by the [Centrifugal fdelta project](https://github.com/centrifugal/fdelta/tree/v0.0.3/internal/cref), including its build headers. All four files come from `internal/cref/fossil/` in the [v0.0.3 Go module release](https://proxy.golang.org/github.com/centrifugal/fdelta/@v/v0.0.3.zip). The release archive has SHA-256 `cb58a34e23b1d6f1615a6926be8cf11e978bb41659add36ebd48686e35bc18db`. Per-file digests appear in `upstream/SHA256SUMS`. The Fossil Simplified BSD license and fdelta module license accompany the files.
