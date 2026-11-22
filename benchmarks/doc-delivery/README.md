# Fetching the Backfold quickstart with a bounded read

A tool reading the first 4,096 response bytes could recover seven of eight selected answers from Backfold’s raw Markdown and one from GitHub’s rendered HTML. Reading both responses in full recovered all eight when the HTML reader kept link destinations. The two endpoints served the same README at the same commit.

## Run it

Use Python 3.9 or later with internet access. The script fetches public documents and prints the exact answers recovered at each byte budget. It requires only the standard library.

```sh
python benchmarks/doc-delivery/check.py
```

[Raw Markdown input](https://raw.githubusercontent.com/pgg898200-ship-it/backfold/d3f3ad5cc8f2b74cd0a67acb1c47ffe513989ef0/README.md) · [GitHub README API, requested with the HTML media type](https://api.github.com/repos/pgg898200-ship-it/backfold/readme?ref=d3f3ad5cc8f2b74cd0a67acb1c47ffe513989ef0) · [Source revision](https://github.com/pgg898200-ship-it/backfold/tree/d3f3ad5cc8f2b74cd0a67acb1c47ffe513989ef0)

## What a reader can recover

The eight questions cover the Python install command, forked example output, retraction method, desktop MCP bundle URL, registry URL, sample path, CSV tool names and measured edit time at 100,000 readings. The script searches for literal answer fragments. Presence means the required strings reached the reader. This test does not evaluate comprehension or whether an assistant chooses the correct answer.

| Body budget per response | Raw Markdown | HTML article with link targets | HTML article text without link targets |
| --- | ---: | ---: | ---: |
| 1,024 bytes | 1/8 | 0/8 | 0/8 |
| 2,048 bytes | 3/8 | 0/8 | 0/8 |
| 4,096 bytes | 7/8 | 1/8 | 1/8 |
| 8,192 bytes | 8/8 | 3/8 | 3/8 |
| Full response | 8/8 | 8/8 | 5/8 |

On November 22, 2026, the Markdown body was 5,253 bytes and the HTML body was 12,749 bytes. Each budget truncates the UTF-8 response body before parsing. HTML readers extract article text from that prefix. The link-aware reader also retains anchor destinations. Byte counts exclude headers and transport compression. GitHub can change its HTML renderer independently of the pinned source. The replay prints response hashes so changes can be identified.

A reader that fetches the whole article and keeps its links pays no answer-coverage penalty in these eight cases. A text-only HTML reader loses the bundle URL, registry URL and example path even with the full document. This is an offline extraction comparison on one release. Referral traffic, reader engagement and installs are unmeasured.

## Finding the MCP setup

The [agent guide](https://pgg898200-ship-it.github.io/backfold/llms.txt) now links directly to the desktop bundle, registry metadata, four-row CSV example and raw Markdown quickstart. The [previous guide](https://raw.githubusercontent.com/pgg898200-ship-it/backfold/d3f3ad5cc8f2b74cd0a67acb1c47ffe513989ef0/llms.txt) sent readers to the repository’s setup section. Its text contained zero of the three MCP destinations. The revised guide contains all three.

The script checks both guides. To inspect a local edit before publishing, run `python benchmarks/doc-delivery/check.py --guide llms.txt`.
