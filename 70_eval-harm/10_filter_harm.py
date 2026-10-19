#!/usr/bin/env python3
"""Keep only the segments of input.jsonl that are in confirmed.jsonl.

    python3 filter_harm.py --confirmed confirmed.fr.jsonl v1.earnings.fr.jsonl > v1.earnings.fr.harm.jsonl
    python3 filter_harm.py --confirmed confirmed.cs.jsonl --confirmed confirmed.fr.jsonl input.jsonl -o out.jsonl

--confirmed: output of only_confirmed.py; can be given more than once.
input:       any JSONL with one segment per line and an "audio" field.

Segments are matched by the audio file name (the directory part of the path is ignored). The lines of
the input that match are written unchanged, in the order of the confirmed file(s) (with several
--confirmed files: first file first). Statistics go to stderr, and the confirmed segments that are not
in the input are listed there.

As a module: match(confirmed_paths, input_path, stats) gives the matched segments with both the
confirmed records and the input records (used by same_harm.py).
"""
import argparse
import json
import os
import sys
from collections import Counter, OrderedDict


def shout(msg):
    print(msg, file=sys.stderr)


def audio_name(rec):
    return os.path.basename(rec.get("audio") or "")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                yield n, line, json.loads(line)


def load_confirmed(paths, stats):
    """OrderedDict {audio name: [confirmed records]}, in the order of the files (first file first)."""
    confirmed = OrderedDict()
    for path in paths:
        for n, _, rec in read_jsonl(path):
            name = audio_name(rec)
            if not name:
                shout(f"WARNING: {path}:{n}: no audio field, skipped")
                stats["confirmed records without audio (skipped)"] += 1
                continue
            confirmed.setdefault(name, []).append(rec)
    stats["confirmed segments"] = len(confirmed)
    return confirmed


def read_input(path, names, stats):
    """{audio name: [(line, record)]} for the input segments whose audio name is in names."""
    found = {}
    for _, line, rec in read_jsonl(path):
        stats["input segments"] += 1
        name = audio_name(rec)
        if name in names:
            found.setdefault(name, []).append((line if line.endswith("\n") else line + "\n", rec))
    return found


def match(confirmed_paths, input_path, stats):
    """[(audio name, [confirmed records], [(input line, input record)])] in the order of the
    confirmed file(s). Confirmed segments missing from the input are reported and left out."""
    confirmed = load_confirmed(confirmed_paths, stats)
    found = read_input(input_path, confirmed, stats)
    stats["confirmed segments not in input"] = 0
    out = []
    for name, recs in confirmed.items():
        if name not in found:
            shout(f"NOT IN INPUT: {name}")
            stats["confirmed segments not in input"] += 1
            continue
        out.append((name, recs, found[name]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmed", action="append", required=True, metavar="FILE",
                    help="confirmed.jsonl, the output of only_confirmed.py (can be repeated)")
    ap.add_argument("input", help="input.jsonl: the segments to filter")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    args = ap.parse_args()

    stats = Counter()
    matched = match(args.confirmed, args.input, stats)
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for _, _, items in matched:
            for line, _ in items:
                out.write(line)
                stats["input segments kept"] += 1
    finally:
        if args.output:
            out.close()
    for k, v in stats.items():
        shout(f"  {k}: {v}")


if __name__ == "__main__":
    main()
