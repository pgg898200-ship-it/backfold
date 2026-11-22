"""Compare exact EventFold bytes with the Fossil SCM reference delta encoder."""
import ctypes as c
import csv
from hashlib import sha256
from pathlib import Path
import os
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from eventfold import split_sse


def verify_source(source):
    for line in (source / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        if sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"upstream digest mismatch for {name}")


def load_fossil(source, temp, cc):
    so = temp / "fossil.so"
    subprocess.run(cc + ["-std=c99", "-O2", "-shared", "-fPIC",
                         "-DFOSSIL_ENABLE_DELTA_CKSUM_TEST=1",
                         str(source / "delta.c"), "-o", str(so)], check=True)
    lib = c.CDLL(str(so))
    void, uint, integer = c.c_void_p, c.c_uint, c.c_int
    lib.delta_create.argtypes = [void, uint, void, uint, void]
    lib.delta_create.restype = integer
    lib.delta_apply.argtypes = [void, integer, void, integer, void]
    lib.delta_apply.restype = integer
    return lib


def fossil_sizes(lib, frames):
    previous = b""
    total = oracle = 0
    for frame in frames:
        old, new = c.create_string_buffer(previous), c.create_string_buffer(frame)
        delta = c.create_string_buffer(len(frame) + 128)
        size = lib.delta_create(old, len(previous), new, len(frame), delta)
        if not 0 < size <= len(delta):
            raise ValueError("Fossil failed to create a delta")
        out = c.create_string_buffer(len(frame) + 1)
        got = lib.delta_apply(old, len(previous), delta, size, out)
        if got != len(frame) or out.raw[:got] != frame:
            raise ValueError("Fossil round-trip failed")
        total += size
        oracle += min(size, len(frame))
        previous = frame
    return total, oracle


def main():
    source = HERE / "upstream"
    verify_source(source)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("cohort", "file", "events", "raw_bytes", "eventfold_bytes",
                     "fossil_delta_bytes", "fossil_best_of_raw_delta_bytes"))
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        cc = shlex.split(os.environ.get("CC", "cc"))
        fossil = load_fossil(source, temp, cc)
        native = str(temp / "eventfold")
        subprocess.run(cc + ["-std=c99", "-O2", str(ROOT / "cli/main.c"), "-o", native],
                       check=True)
        for cohort, path in [("development", ROOT / "fixtures"),
                             ("holdout", ROOT / "fixtures/holdout")]:
            for f in sorted(path.glob("*.sse")):
                original = f.read_bytes()
                frames = list(split_sse([original]))
                if b"".join(frames) != original:
                    raise ValueError(f"frame splitter failed on {f.name}")
                delta, oracle = fossil_sizes(fossil, frames)
                packed = subprocess.check_output([native, "pack"], input=original)
                if subprocess.check_output([native, "unpack"], input=packed) != original:
                    raise ValueError(f"EventFold round-trip failed on {f.name}")
                writer.writerow((cohort, f.name, len(frames), len(original),
                                 len(packed), delta, oracle))


if __name__ == "__main__":
    main()
