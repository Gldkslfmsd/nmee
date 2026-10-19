#!/usr/bin/env python3
"""Skim JSONL segments: drop bulky fields, optionally keep only some languages.

Usage:
    cat in.jsonl | python skim.py > out.jsonl
    cat in.jsonl | python skim.py --lan cs en > out.jsonl
"""
import argparse
import json
import sys

DROP_FIELDS = ("words", "alignment", "gold_meta", "segmented_by")
# Fields keyed by system name, which are filtered by --lan.
PER_SYSTEM_FIELDS = ("systems_info", "text")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lan", nargs="+", metavar="LAN", default=None,
                        help="keep only systems whose language code is in this list "
                             "(e.g. --lan cs en). Default: keep all systems.")
    parser.add_argument("--remove-systems", nargs="+", metavar="S", default=[],
                        help="remove these systems. Default: do not remove anything.")
    args = parser.parse_args()
    keep_lans = set(args.lan) if args.lan else None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        j = json.loads(line)

        for field in DROP_FIELDS:
            j.pop(field, None)

        if keep_lans is not None:
            keep_systems = {
                name for name, info in j.get("systems_info", {}).items()
                if info.get("lan") in keep_lans and name not in args.remove_systems
            }
            for field in PER_SYSTEM_FIELDS:
                if field in j:
                    j[field] = {k: v for k, v in j[field].items() if k in keep_systems}

        print(json.dumps(j, ensure_ascii=False))


if __name__ == "__main__":
    main()
