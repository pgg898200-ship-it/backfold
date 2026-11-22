"""Build the reproducible Backfold 0.3.0 bundle from verified release binaries."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
ASSETS = {
    "backfold-aarch64-apple-darwin": "973581d01c16c2bb68db4edff18d7c9ae28945ba9d587fd9524cd929e094767f",
    "backfold-x86_64-pc-windows-msvc.exe": "2bcbf6b5ca02c927376a05b8f2e8b59d0bc0c446852fc25bfcdcd2633202879b",
    "backfold-x86_64-unknown-linux-gnu": "f10da9adafc79195b4304b9e561e0537fb537231966a075a0a47211cbd8f4e8a",
}
RELEASE = "https://github.com/pgg898200-ship-it/backfold/releases/download/v0.3.0/"


def write_entry(archive, name, data, mode):
    # Stored entries and fixed timestamps produce identical bytes on all three hosts.
    entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.create_system = 3
    entry.external_attr = (0o100000 | mode) << 16
    archive.writestr(entry, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, help="Use previously downloaded release files")
    parser.add_argument("--output", type=Path, default=Path("backfold-0.3.0.mcpb"))
    args = parser.parse_args()
    sources = {}
    for name, expected in ASSETS.items():
        if args.assets:
            data = (args.assets / name).read_bytes()
        else:
            with urlopen(RELEASE + name, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Release checksum mismatch for {name}")
        sources[name] = data

    # Normalize checkout line endings, including on Windows.
    manifest = (ROOT / "integrations/mcpb/manifest.json").read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8").rstrip() + "\n"
    with ZipFile(args.output, "w") as archive:
        write_entry(archive, "manifest.json", manifest.encode("utf-8"), 0o644)
        write_entry(archive, "LICENSE", license_text.encode("utf-8"), 0o644)
        for name, data in sources.items():
            write_entry(archive, f"bin/{name}", data, 0o755)
    print(f"{hashlib.sha256(args.output.read_bytes()).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
