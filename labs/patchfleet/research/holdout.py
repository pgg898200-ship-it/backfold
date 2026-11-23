"""Frozen-format evaluation on the 460 Verified tasks outside the 40-task design set."""

import concurrent.futures
import json
import lzma
from pathlib import Path
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bundle
from swe import download, digest, SUBMISSIONS, HERE


def main(cache, output, split):
    pins = json.loads((HERE / "swe.sources.json").read_text())
    cases = (
        pins["cases"]
        if split == "design"
        else json.loads((HERE / "swe.holdout.json").read_text())
    )
    tracks = []
    for name, af in SUBMISSIONS.items():
        url = f"https://swe-bench-submissions.s3.amazonaws.com/{af}/{name}/all_preds.jsonl"
        raw = download(url, cache / (name + "-all_preds.jsonl"))
        if digest(raw) != pins["archives"][name]:
            raise ValueError("Changed archive")
        tracks.append(
            {
                r["instance_id"]: (r.get("model_patch") or "").encode()
                for r in map(json.loads, raw.splitlines())
            }
        )

    def read(repo, base, p):
        return download(
            f"https://raw.githubusercontent.com/{repo}/{base}/{quote(p)}",
            cache / "base" / base / digest(p.encode()),
            True,
        )

    groups = [(*cases[key], [t.get(key, b"") for t in tracks]) for key in sorted(cases)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=18) as ex:
        parts = list(ex.map(lambda g: bundle.collect([g], read), groups))
    manifest, rows, basis = [], [], bytearray()
    for m, r, b in parts:
        manifest.extend(m)
        rows.extend(r)
        basis.extend(b)
    archive, sizes = bundle.encode(rows, manifest, bytes(basis))
    copies = bundle.decode(archive, lambda m: bundle.dictionary(m, read))
    if copies != rows:
        raise ValueError("Submitted bytes mismatch")
    raw = bundle.pack_rows(rows)
    out = dict(
        split=split,
        cases=len(cases),
        records=len(rows),
        nonempty=sum(bool(r) for r in rows),
        framed_input_bytes=len(raw),
        source_bytes=len(basis),
        sum_codec_and_cli_xz9=len(
            lzma.compress((HERE.parent / "bundle.py").read_bytes(), preset=9)
        ),
        verified_sha256=digest(raw),
        archive_bytes=len(archive),
        mode=bundle.MODES[archive[4]],
        charged_archive_bytes=sizes,
        plain_xz9=len(lzma.compress(raw, preset=9)),
        raw_zstd19_only=len(bundle.zstd(raw)),
        archive_sha256=digest(archive),
    )
    output.write_text(json.dumps(out, indent=2) + "\n")
    # The standalone archive is a local reproduction output containing the submitters' patches.
    (cache / f"{split}.pfw").write_bytes(archive)
    (cache / f"{split}.base").write_bytes(basis)
    (cache / f"{split}.raw").write_bytes(raw)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
