# Benchmark corpus

Each `.sse` file is one independent, ordered stream. The corpus contains 1,101 frames in 10 streams across 7 providers, totaling 323,698 bytes. It covers text deltas, tool arguments, refusal, fallback, and mixed metadata. Groq and OpenAI chat text account for 966 of the 1,101 frames. The mix is a convenience sample from public SDK tests. It has no measured production arrival times, traffic weights, or transport packet boundaries.

## Sources and changes

The three `anthropic-stream-*.sse` streams are verbatim copies from the official Anthropic TypeScript SDK's fallback tests, with local filenames prefixed for clarity. They include the original whitespace and event labels. [Upstream source](https://github.com/anthropics/anthropic-sdk-typescript/tree/3edf5ebe90b1b40a5ed3b065139b3c7727c88b01/tests/fixtures/fable-fallback)

The other seven streams carry payloads from Vercel AI SDK tests. The upstream files each store one JSON payload per line. We preserved every payload byte and supplied LF framing. For each source line we emitted `data: `, the payload, and two LF bytes. For the OpenAI Responses custom tool fixture we also prefixed each frame with `event: `, its JSON `type`, and one LF. The streams end after the last source payload. The framing represents one replay convention. [Upstream packages](https://github.com/vercel/ai/tree/d55f3cfc456d1b7bc7db3a4b70fbf602ddc8647d/packages)

The [manifest](manifest.json) records the source repository and revision, path, Git blob hash, byte count, framing, final SHA-256, and frame count for every stream. To check all bundled fixtures, run this from the project root.

```sh
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

fixtures = Path("fixtures")
for row in json.loads((fixtures / "manifest.json").read_text()):
    data = (fixtures / row["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == row["sse_sha256"], row["file"]
print("All fixture hashes match")
PY
```

## Notices

The Anthropic fixtures come from the Anthropic TypeScript SDK, under [Apache License 2.0](licenses/Apache-2.0.txt). The repository also includes a [historical MIT license and attribution](licenses/Anthropic-LICENSE-MIT.txt).

The adapted Vercel fixtures are copyright 2023 Vercel, Inc., licensed under the [Apache License 2.0](licenses/Apache-2.0.txt). The [original notice](licenses/Vercel-LICENSE.txt) accompanies the adapted files.
