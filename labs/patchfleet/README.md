# Patchfleet

**One 1,928-byte file replays the NumPy 2 codemod across 24 upstream release trees.** Give it a pinned Git checkout. It writes the migrated objects, verifies the input/output tree pair and prints the new tree ID. Your branch, index and working files stay where they are.

```sh
git clone --depth 1 --branch networkx-2.8.8 https://github.com/networkx/networkx.git example
git clone --depth 1 https://github.com/pgg898200-ship-it/backfold.git
cd backfold/labs/patchfleet
repo=../../../example
tree=$(python3 fleet.py artifacts/numpy2.fleet.xz "$repo" HEAD)
git -C "$repo" diff HEAD "$tree" --stat
```

For a new revision, the incremental compiler uses the installed Ruff reference on previously unseen source revisions. Its persistent SQLite cache keys fixes and clean files by source blob and path.

```sh
python3 -m pip install ruff==0.11.13
python3 incremental.py "$repo" HEAD fleet.sqlite3
```

## What fits in the capsule

The corpus covers four release tags each for librosa, NetworkX, nibabel, scikit-image, seaborn and xarray. We ran the NumPy team's recommended NPY201 rule, with safe fixes enabled, on each tracked regular Python file using isolated settings. That produced 96 changed file instances across 7,266 files and 81,175,718 input bytes. The edits reduce to 29 distinct programs. Every original non-Python Git object, filename and file mode survives in the output tree. Several releases already needed zero edits.

Git already supplies source blob identities. A short prefix selects an edit program, and a SHA-256 commitment over the full input and output tree IDs verifies the result. This permits small selectors without trusting them for correctness. The verifier checks against the published manifest. The capsule SHA-256 is `afe736b2eb36c1db01e7164eb7915e34ffbdfaf643eded477c758707b156c717`. Unrecognized trees are handled by the separate compiler.

| Transfer, all 24 known snapshots | Bytes |
| --- | ---: |
| Concatenated Git diffs, zstd -19 | 9,132 |
| Minimal zero-context diffs, stripped headers, zstd -19 | 2,667 |
| Direct offset-and-literal manifest, zstd -19 | 2,398 |
| Patchfleet capsule | **1,928** |

The stripped patches were applied using stock Git and each output tree was checked. Their byte count excludes release framing. The capsule includes release selection and output certificates. The Python decoder ships separately. Charging its compressed source changes the single-download comparison in favor of stripped Git patches. Ruff already installed at both ends can regenerate this public rule from a short command. The capsule's use case is distributing an already computed result with offline replay. Timing, accuracy and package-wide compatibility are separate quantities.

[Replay an upstream file live](https://pgg898200-ship-it.github.io/backfold/patchfleet/)

## Independent evidence

All 24 reconstruction targets were computed by running unmodified Ruff 0.11.13 and Git. Sequential incremental compilation produced the same 24 trees and parsed 3,970 of 7,266 file instances. On the six newest releases it parsed 1,063 of 1,874. Reusing old changed blobs alone recovered four of eleven required edits on these held-out releases. Compilation of unseen inputs matters.

Four original NetworkX 2.8.8 test modules on NumPy 2.2.6 went from 57/142 passing to 140/142 after replay. The remaining two call a removed alias through a dynamically imported module. The detailed result lists both. Across the full NetworkX suite, 106 tests went from failing to passing. The migrated tree had 5,133 passes, 9 failures and 17 skips, with zero pass-to-failure transitions. [Full suite output](research/upstream-full.local.json). Across four separately selected historical migration changes, the pinned safe rule reproduced 40 of 59 modified Python files byte for byte. This is a check against maintainer decisions, including their different choices and changes beyond the rule's scope.

## Reproduce

```sh
python3 -m pip install ruff==0.11.13
python3 research/test_codec.py
python3 fetch.py /tmp/fleet-work
python3 build.py /tmp/fleet-work /tmp/fleet-rebuilt
cmp artifacts/numpy2.fleet.xz /tmp/fleet-rebuilt/numpy2.fleet.xz
python3 research/incremental_replay.py /tmp/fleet-work /tmp/incremental.json
```

Git and the zstd command are required for the benchmark. The offline decoder requires only Git and Python 3.10+. Source tags and immutable commit IDs are in [sources.lock.json](sources.lock.json). Per-release results, byte counts, codec overhead, and the file-holdout breakdown are in [results.json](artifacts/results.json). The [reproduction workflow](../../.github/workflows/patchfleet-reproduce.yml) rebuilds from the original six projects on a clean Ubuntu runner. The [public verification run](https://github.com/pgg898200-ship-it/backfold/actions/runs/45130021432) matched the archive bytes and 128 Git trees from the published measurements. Its workflow and benchmark design are maintained in this repository.

The [research directory](research) includes the early-start matrix and scalar Laplace attention prototypes, externally authored tests, historical edits and a separate lossless coding-agent patch benchmark. These measure different outputs. Benchmarks of the scalar head use its own normalized kernel. The scripts do not fit or evaluate a repository-language model.

## Keep a migration current on every commit

```sh
python3 history_compile.py /path/to/repo history.sqlite3 HEAD~2 HEAD~1 HEAD
```

The compiler fingerprints subtrees at their repository paths. When an agent forks a known snapshot, unchanged directories return their verified Git objects. On new directories it descends, reuses previously compiled files and submits the remaining source to Ruff in batches. If Git has collected a cached output tree, it rebuilds it. The original checkout and refs are preserved.

On 128 consecutive scikit-image first-parent commits the compiler handed **914 of 68,500 Python file instances** to Ruff. All 128 whole-repository outputs matched a separate Git worktree running stock Ruff with its own persistent cache. Eighty-four snapshots needed edits. A second baseline used Git to select changed paths plus the paths previously fixed. It submitted 1,080 files and also matched every output tree. Both baselines include the cost of producing a Git output tree. The corpus, contiguous endpoints, content checks and local clock readings are in [commit-stream measurements](research/commits.local.json) and [changed-path measurements](research/git-parent.local.json). Reproduction timings depend on storage, OS cache and the runner.

## Archive an agent fleet's actual patches

[Download Patchfleet Archive 1.0.0](https://pgg898200-ship-it.github.io/backfold/patchfleet/releases/patchfleet-archive-1.0.0.zip) · [Verify the ZIP and view release measurements](https://pgg898200-ship-it.github.io/backfold/patchfleet/#archive)

**1,380 held-out coding-agent outputs fit in 1,089,297 bytes.** The input is the exact submitted patch text from three public SWE-bench Verified runs. The receiver already has the source repositories. The output reconstructs every patch byte, including generated files, comments, hunk headers and empty submissions. It preserves record order. Submitter metadata can be kept separately.

The encoder separates each patch line's leading diff marker from its body, restoring source code to a representation a stock delta compressor can match. It also tries the raw stream, the standard source dictionary, and source rendered as diff lines. The selected archive includes its mode, source lookup manifest, and a SHA-256 of the original records. The receiver supplies 30,102,631 bytes of referenced source for this workload. The manifest identifies every required commit and path. The compressed stand-alone Python tool adds roughly 3 KB to a first download. The format targets teams with local Git clones.

| Held-out transfer | Bytes |
| --- | ---: |
| Stock zstd -19 | 1,754,336 |
| Brotli quality 11 | 1,705,074 |
| xdelta3 then xz -9 | 1,673,532 |
| Source-aware zstd -22, 128 MiB window | 1,357,083 |
| Patchfleet level 12, with source manifest and digest | 1,225,677 |
| Patchfleet default level 19 | 1,089,297 |
| Patchfleet level 22 | **1,046,922** |

The 40 design tasks were selected by hashing the 500 public instance IDs. The other 460 constitute the holdout. In the held-out set 1,349 records contained edits. A separate run per submission also reduced bytes against the corresponding stock zstd -19 source-delta wrapper for each of the three agents. [Held-out measurements](research/heldout.json) · [Other codecs](research/strong-codecs.json) · [Per-project and per-issue packing](research/partitions.json) · [Results by submitter](research/strata.json).

To package your own outputs, provide a JSON list in execution order. Use full Git commit IDs, local patch paths relative to the manifest, and clones at `clones/owner/repository`. The manifest can contain many commits and repositories.

```json
[{"repo":"networkx/networkx", "commit":"9256ef670730b741105a9264197353004bd6719f", "patches":["agent-01.patch","agent-02.patch"]}]
```

```sh
python3 bundle.py pack clones tasks.json work.pfw
python3 bundle.py unpack clones work.pfw restored
```

Version 1.0 selects columns directly when referenced source is available. Level 12 favors encoding speed. Level 22 uses a 128 MiB window for compact transfer. All three settings restore the same records in the original PFW1 format. Use `--mode auto` to search four representations or `--mode raw` to pack without a source checkout. [Release profile results](research/release-1.0.0.local.json).

In the original timing study on the 460-task holdout, choosing the column representation directly produced exactly the same archive in 12.61 seconds. Searching all four modes took 42.47 seconds. Stock source-aware zstd -22 took 18.84 seconds. Decode medians were 0.161 seconds for Patchfleet and 0.108 seconds for stock zstd. These are three local CPU runs with the source dictionary already assembled. Fetch and dictionary-discovery time are additional. [Codec costs](research/codec-cost.local.json) · [Single-pass encoder](research/codec-fixed.local.json).

Restore writes numbered patch files to a new directory. The packer checks every byte before publishing an archive. For byte-exact benchmark replication, use zstd 1.5.7. To rerun the full held-out comparison:

```sh
python3 research/holdout.py /tmp/agent-data /tmp/heldout.json heldout
python3 research/partitions.py /tmp/agent-data /tmp/partitions.json
python3 research/release_profiles.py /tmp/agent-data
```

Data provenance: [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) and the [official submissions archive](https://github.com/SWE-bench/experiments). The external task grades belong to the original submissions. Patchfleet's comparisons concern recoverability and transfer size. No leaderboard rank is claimed.

## Early-start inference

The alternative matrix prototype starts a down-projection while the preceding activation is provisional. Linearity permits an exact correction `W @ (h - preview)` in real arithmetic. It saves work when enough entries of the correction are zero.

We tried it in the first 512→2048→512 MLP of the released EleutherAI Pythia-70M checkpoint, using its tokenizer and 1,536 tokens of upstream code. Even with **511 of 512 input channels already present**, arrival of the last channel changed **94.2%** of the fp16-rounded intermediate activations and 99.99% in fp32. On this checkpoint and preview scheme, an exact sparse correction has little work to skip. [Checkpoint, immutable weight hashes and reproduction](research/checkpoint_preview.py) · [Measured correction densities](research/checkpoint.json).

The indexed scalar Laplace attention head also runs independently. Its measured numerical agreement is for that kernel. Its training quality on code retrieval and compatibility with conventional softmax attention remain open. [Operator prototypes](research/operators.py).
