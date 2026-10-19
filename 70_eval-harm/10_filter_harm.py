#!/usr/bin/env python3
"""Keep only the segments of input.jsonl that are in confirmed.jsonl.

    python3 filter_harm.py --confirmed confirmed.fr.jsonl v1.earnings.fr.jsonl > v1.earnings.fr.harm.jsonl
    python3 filter_harm.py --confirmed confirmed.cs.jsonl --confirmed confirmed.fr.jsonl input.jsonl -o out.jsonl

--confirmed: output of only_confirmed.py; can be given more than once.
input:       any JSONL with one segment per line and an "audio" field.

Segments are matched by the audio file name (the directory part of the path is ignored). The lines of
the input that match are written unchanged, in the order of the confirmed file(s) (with several
--confirmed files: first file first). Statistics go to stderr, and
the confirmed segments that are not in the input are listed there.
"""
import argparse
import json
import os
import sys
from collections import Counter


def shout(msg):
    print(msg, file=sys.stderr)


def audio_name(rec):
    return os.path.basename(rec.get("audio") or "")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield n, line, json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmed", action="append", required=True, metavar="FILE",
                    help="confirmed.jsonl, the output of only_confirmed.py (can be repeated)")
    ap.add_argument("input", help="input.jsonl: the segments to filter")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    args = ap.parse_args()

    stats = Counter()
    confirmed = []  # audio names in the order of the confirmed file(s), without duplicates
    seen = set()
    for path in args.confirmed:
        for n, _, rec in read_jsonl(path):
            name = audio_name(rec)
            if not name:
                shout(f"WARNING: {path}:{n}: no audio field, skipped")
                stats["confirmed records without audio (skipped)"] += 1
                continue
            if name not in seen:
                seen.add(name)
                confirmed.append(name)
    stats["confirmed segments"] = len(confirmed)

    lines = {}  # audio name -> lines of the input with it (normally one)
    for n, line, rec in read_jsonl(args.input):
        stats["input segments"] += 1
        name = audio_name(rec)
        if name in seen:
            lines.setdefault(name, []).append(line if line.endswith("\n") else line + "\n")

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    missing = 0
    try:
        for name in confirmed:
            if name not in lines:
                shout(f"NOT IN INPUT: {name}")
                missing += 1
                continue
            for line in lines[name]:
                out.write(line)
                stats["input segments kept"] += 1
    finally:
        if args.output:
            out.close()
    stats["confirmed segments not in input"] = missing
    for k, v in stats.items():
        shout(f"  {k}: {v}")


if __name__ == "__main__":
    main()
