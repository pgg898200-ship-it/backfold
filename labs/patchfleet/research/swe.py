"""Rebuild a preselected sample of three public SWE-bench prediction archives."""

import argparse
import concurrent.futures
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen
from patchwire import decode, encode, length, lines, source_path

SUBMISSIONS = {
    "20240402_sweagent_gpt4": "verified",
    "20250225_sweagent_claude-3-7-sonnet": "verified",
    "20250726_mini-v1.0.0_claude-sonnet-4-20250514": "bash-only",
}
HERE = Path(__file__).resolve().parent


def digest(b):
    return hashlib.sha256(b).hexdigest()


def download(url, path, missing=False):
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as res:
                data = res.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return data
        except HTTPError as err:
            if err.code == 404 and missing:
                return b""
            if attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise
        time.sleep(1 + attempt)


def main(cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    pins = (
        json.loads((HERE / "swe.sources.json").read_text())
        if (HERE / "swe.sources.json").exists()
        else {}
    )
    tracks, source_hash = {}, {}
    for name, af in SUBMISSIONS.items():
        url = f"https://swe-bench-submissions.s3.amazonaws.com/{af}/{name}/all_preds.jsonl"
        data = download(url, cache / (name + "-all_preds.jsonl"))
        source_hash[name] = digest(data)
        if pins and source_hash[name] != pins["archives"][name]:
            raise ValueError("Changed submission " + name)
        tracks[name] = {
            r["instance_id"]: (r.get("model_patch") or "").encode()
            for r in map(json.loads, data.splitlines())
        }
    if pins:
        cases = pins["cases"]
    else:
        rows = {}

        def page(i):
            url = f"https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Verified&config=default&split=test&offset={i}&length=100"
            return json.loads(download(url, cache / f"hf-{i}.json"))["rows"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            for page_rows in pool.map(page, range(0, 500, 100)):
                for row in page_rows:
                    r = row["row"]
                    rows[r["instance_id"]] = [r["repo"], r["base_commit"]]
        if len(rows) != 500:
            raise ValueError("Incomplete Verified split")
        cases = {
            k: rows[k] for k in sorted(rows, key=lambda i: digest(i.encode()))[:40]
        }
        (cache / "swe.sources.json").write_text(
            json.dumps(dict(archives=source_hash, cases=cases), indent=2) + "\n"
        )
    sources = {}

    def fetch_case(item):
        case, (repo, base) = item
        names = {
            p
            for track in tracks.values()
            for line in lines(track.get(case, b""))
            if (p := source_path(line))
        }
        ans = {}
        for p in sorted(names):
            url = f"https://raw.githubusercontent.com/{repo}/{base}/{quote(p)}"
            data = download(url, cache / "base" / base / digest(p.encode()), True)
            if data:
                ans[p] = lines(data)
        return case, ans

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        sources.update(pool.map(fetch_case, cases.items()))
    raw, pack, base_dump = bytearray(), bytearray(), bytearray()
    nonempty, count = {}, 0
    for case in sorted(cases):
        for p in sorted(sources[case]):
            base_dump.extend(b"".join(sources[case][p]))
        for name, track in tracks.items():
            original = track.get(case, b"")
            compact = encode(original, sources[case])
            if decode(compact, sources[case]) != original:
                raise ValueError(f"Round trip failed {name}/{case}")
            raw.extend(length(len(original)) + original)
            pack.extend(length(len(compact)) + compact)
            count += 1
            nonempty[name] = nonempty.get(name, 0) + bool(original)

    def zstd(data, *extra):
        return subprocess.check_output(
            ["zstd", "-q", "-19", *extra, "--stdout"], input=data
        )

    with tempfile.TemporaryDirectory() as td:
        basis = Path(td) / "base"
        basis.write_bytes(base_dump)
        delta = zstd(raw, f"--patch-from={basis}", f"--stream-size={len(raw)}")
        restored = subprocess.check_output(
            ["zstd", "-q", "-d", f"--patch-from={basis}", "--stdout"], input=delta
        )
        assert restored == raw
    result = dict(
        cases=len(cases),
        records=count,
        nonempty_per_submission=nonempty,
        base_bytes=len(base_dump),
        framed_original_bytes=len(raw),
        roundtrip_sha256=digest(raw),
        zero_mismatches=True,
        diff_zstd19=len(zstd(raw)),
        wire_zstd19=len(zstd(pack)),
        diff_xz9=len(lzma.compress(raw, preset=9)),
        wire_xz9=len(lzma.compress(pack, preset=9)),
        zstd_patch_from_source_bytes=len(delta),
        wire_decoder_and_encoder_xz9=len(
            lzma.compress((HERE / "patchwire.py").read_bytes(), preset=9)
        ),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    main(args.cache, args.output)
