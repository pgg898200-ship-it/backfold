# Choose a PGG tool

[Browse the full PGG Engineering directory](https://github.com/pgg898200-ship-it/streamsieve/blob/main/PRODUCTS.md) for all products, their source, and their place in the stack. Use the comparisons below when two tools seem to fit the same job.

## Which rank-fusion tool fits my search service?

The deciding factor is what a retriever may do after it sends its first results.

| What happens next | Use | What the consumer gets |
| --- | --- | --- |
| A worker can replace its list or withdraw its contribution. | [RankPatch](https://github.com/pgg898200-ship-it/rankpatch) | A current fused top *k* after each list revision. [Edit a list](https://huggingface.co/spaces/pgg-engineering/rankpatch) |
| Each source appends stable pages. Documents may overlap across sources, and sources can have different weights. | [RankSeal](https://huggingface.co/spaces/pgg-engineering/rankseal) | A proof when the ordered weighted reciprocal-rank-fusion result is fixed. [Audit two ranked lists](https://pgg-engineering-rankseal.static.hf.space/workbench.html) |
| Each source appends stable pages. The coordinator knows which shards have disjoint document IDs. | [RankFence](https://github.com/pgg898200-ship-it/streamsieve/tree/rankfence-v0.1.0) | The global RRF result under the component-cursor rank convention. The coordinator can stop reading a shard when it cannot change the winners. [Replay a shard map](https://rankfence-pgg.tancurl3.chatgpt.site/workbench.html) |

## How does an offline reader catch up?

Suppose a worker publishes capacity 8, then 7, then 6 while a reader sleeps. A latest-state consumer can apply one row containing 6. A consumer that bills for each transition needs the event history.

| How readers consume the updates | Use | Where it runs |
| --- | --- | --- |
| Each reader polls independently for keys that changed after its cursor. | [Afterkey](https://github.com/pgg898200-ship-it/streamsieve/tree/main/afterkey) | A C++17 publisher serving replacement values and deletions. [Pause a reader](https://huggingface.co/spaces/pgg-engineering/afterkey) |
| A C99 or Python reader needs bounded pages over dense integer keys and a cursor to resume. | [Tailstate](https://github.com/pgg898200-ship-it/streamsieve/tree/afterkey-tailstate-0.1.0/afterkey/companions/tailstate) | The publisher keeps one current entry per key. [Run the SQLite reconnect example](https://github.com/pgg898200-ship-it/streamsieve/blob/afterkey-tailstate-0.1.0/afterkey/companions/tailstate/examples/reconnect.py) |
| The publisher serves many readers together at the end of a tick. | [SinceKit](https://huggingface.co/spaces/pgg-engineering/sincekit) | A batch planner builds one changed-slot array and gives each reader its own prefix length. [Compare reader checkpoints](https://huggingface.co/spaces/pgg-engineering/sincekit) |

## What should wake the dispatcher?

| Reason to wake | Event the tool delivers | Try it |
| --- | --- | --- |
| A feed goes quiet longer than a reader’s deadline. | **Pulsefold** identifies affected heartbeat subscribers. | [Replay heartbeats](https://huggingface.co/spaces/pgg-engineering/pulsefold) |
| A rolling count reaches a threshold, or drops below it when old arrivals expire. | **KthWake** emits changed rules and the next expiry for the host timer. | [Run the three-arrival example](https://github.com/pgg898200-ship-it/chronosieve/tree/releases/kthwake-0.1.0/companions/kthwake) |
| Numeric conditions change after a batch updates several signals. | **Guardshift** emits the IDs that entered or exited their conditions. | [Change a signal](https://huggingface.co/spaces/pgg-engineering/guardshift) |
| A moving point gains or loses membership in a fixed rectangular zone. | **EdgeChime** emits the zone IDs entered or left between two positions. | [Replay labeled moves](https://huggingface.co/datasets/pgg-engineering/edgechime-bench) |

## Do I need a prompt diff or a cache size?

**[CachePin](https://pgg898200-ship-it.github.io/cachepin/#compare)** sits on the request path. Give it two requests to inspect where their shared byte prefix ends and what its canonicalization changes. **[Prefix Atlas](https://huggingface.co/spaces/pgg-engineering/prefix-atlas)** runs offline. Give it a trace to estimate how many complete KV pages a serial LRU cache needs for a target prefix hit rate.
