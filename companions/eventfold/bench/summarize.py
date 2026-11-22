#!/usr/bin/env python3
"""Print totals for each independent corpus and synthetic workload."""
import csv
import sys
from collections import defaultdict

groups = defaultdict(lambda: defaultdict(list))
with open(sys.argv[1], newline="") as source:
    for row in csv.DictReader(source):
        suite = row["workload"].split("/")[0]
        groups[suite][row["codec"]].append(row)

print("suite,codec,streams,events,raw_bytes,wire_bytes,wire_percent,weighted_encode_ns,weighted_decode_ns")
for suite, by_codec in groups.items():
    for codec, rows in by_codec.items():
        events = sum(int(r["events"]) for r in rows)
        raw = sum(int(r["raw_bytes"]) for r in rows)
        wire = sum(int(r["wire_bytes"]) for r in rows)

        def weighted(key):
            return sum(float(r[key])*int(r["events"]) for r in rows)/events

        print(f"{suite},{codec},{len(rows)},{events},{raw},{wire},{100*wire/raw:.2f},"
              f"{weighted('encode_ns_per_event'):.2f},{weighted('decode_ns_per_event'):.2f}")
