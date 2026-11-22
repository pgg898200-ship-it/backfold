"""Streaming EFL1 filter, reading stdin and writing stdout."""
import argparse
import json
import sys

from . import DEFAULT_MAX_FRAME_SIZE, pack_stream, unpack_stream


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "unpack", "compare"))
    parser.add_argument("--max-frame", type=int, default=DEFAULT_MAX_FRAME_SIZE,
                        metavar="BYTES", help="maximum decoded frame size")
    options = parser.parse_args()
    try:
        if options.action == "compare":
            from .compare import compare_stream
            print(json.dumps(compare_stream(sys.stdin.buffer, max_frame_size=options.max_frame), indent=2))
            return 0
        convert = pack_stream if options.action == "pack" else unpack_stream
        convert(sys.stdin.buffer, sys.stdout.buffer, max_frame_size=options.max_frame)
    except (OSError, ValueError) as error:
        print(f"eventfold: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
