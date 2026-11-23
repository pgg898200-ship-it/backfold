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

Git already supplies source blob identities. A short prefix selects an edit program, and a SHA-256 commitment over the full input and output tree IDs verifies the result. This permits small selectors without trusting them for correctness. A modified input or a damaged program cannot silently return an unverified tree. Unrecognized trees are handled by the separate compiler.

| Transfer, all 24 known snapshots | Bytes |
| --- | ---: |
| Concatenated Git diffs, zstd -19 | 9,132 |
| Minimal zero-context diffs, stripped headers, zstd -19 | 2,667 |
| Direct offset-and-literal manifest, zstd -19 | 2,398 |
| Patchfleet capsule | **1,928** |

The stripped patches were applied using stock Git and each output tree was checked. Their byte count excludes release framing. The capsule includes release selection and output certificates. The Python decoder ships separately. Charging its compressed source changes the single-download comparison in favor of stripped Git patches. Ruff already installed at both ends can regenerate this public rule from a short command. The capsule's use case is distributing an already computed result with offline replay. Timing, accuracy and package-wide compatibility are separate quantities.

## Independent evidence

All 24 reconstruction targets were computed by running unmodified Ruff 0.11.13 and Git. Sequential incremental compilation produced the same 24 trees and parsed 3,970 of 7,266 file instances. On the six newest releases it parsed 1,063 of 1,874. Reusing old changed blobs alone recovered four of eleven required edits on these held-out releases. Compilation of unseen inputs matters.

Four original NetworkX 2.8.8 test modules on NumPy 2.2.6 went from 57/142 passing to 140/142 after replay. The remaining two call a removed alias through a dynamically imported module. The detailed result lists both. Across four separately selected historical migration changes, the pinned safe rule reproduced 40 of 59 modified Python files byte for byte. This is a check against maintainer decisions, including their different choices and changes beyond the rule's scope.

## Reproduce

```sh
python3 -m pip install ruff==0.11.13
python3 research/test_codec.py
python3 fetch.py /tmp/fleet-work
python3 build.py /tmp/fleet-work /tmp/fleet-rebuilt
cmp artifacts/numpy2.fleet.xz /tmp/fleet-rebuilt/numpy2.fleet.xz
python3 research/incremental_replay.py /tmp/fleet-work /tmp/incremental.json
```

Git and the zstd command are required for the benchmark. The offline decoder requires only Git and Python 3.10+. Source tags and immutable commit IDs are in [sources.lock.json](sources.lock.json). Per-release results, byte counts, codec overhead, and the file-holdout breakdown are in [results.json](artifacts/results.json). The [reproduction workflow](../../.github/workflows/patchfleet-reproduce.yml) rebuilds from the original six projects on a clean Ubuntu runner.

The [research directory](research) includes the early-start matrix and scalar Laplace attention prototypes, externally authored tests, historical edits and a separate lossless coding-agent patch benchmark. These measure different outputs. Benchmarks of the scalar head use its own normalized kernel. The scripts do not fit or evaluate a repository-language model.
