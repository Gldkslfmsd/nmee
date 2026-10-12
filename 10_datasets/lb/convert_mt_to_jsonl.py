#!/usr/bin/env python3
"""Convert MT output JSONL ({"id": ..., "text": ...} per line) to the segment
JSONL format ({"segment": ...} per line). Timing and word fields are omitted
because the MT input has none.

Usage:
    python convert_mt_to_jsonl.py input.jsonl output.jsonl
    python convert_mt_to_jsonl.py input.jsonl output.jsonl --keep-id
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input MT JSONL file")
    ap.add_argument("output", help="output JSONL file")
    ap.add_argument("--keep-id", action="store_true",
                    help="keep the input 'id' field on each output line")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out = {"segment": rec.get("text", "").strip()}
            if args.keep_id and "id" in rec:
                out["id"] = rec["id"]
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
