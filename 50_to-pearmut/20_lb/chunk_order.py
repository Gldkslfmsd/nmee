#!/usr/bin/env python3
"""
Split long documents (e.g. parliament recordings) into virtual time chunks
and reorder the segments chunk-wise.

Each segment is assigned to chunk  floor(beg / chunk_len)  of its document.
Segments inside a chunk keep their original (time) order.

Modes:
  * --chunk-order FILE does not exist  -> shuffle chunks randomly (--seed),
                                          write the order to FILE
  * --chunk-order FILE exists          -> reorder chunks as listed in FILE
    (chunks absent from FILE are dropped with a warning, so the TSV can also
     be used to select a subset)
  * --shuffle forces a new random order and overwrites FILE.

Usage:
  python chunk_order.py input.jsonl -o output.jsonl --chunk-order order.tsv [--seed 42]
"""

import argparse
import json
import os
import random
import sys
from collections import OrderedDict

TSV_HEADER = ["order", "chunk_id", "document", "chunk_index", "beg", "end", "n_segments"]


def chunk_id(doc, idx):
    return f"{doc}#{idx:03d}"


def read_jsonl(path):
    f = sys.stdin if path == "-" else open(path, encoding="utf-8")
    with f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: invalid JSON: {e}")


def build_chunks(segments, chunk_len):
    """Return OrderedDict chunk_id -> dict(document, index, segments)."""
    chunks = OrderedDict()
    for seg in segments:
        doc = seg["document"]
        idx = int(float(seg["beg"]) // chunk_len)
        cid = chunk_id(doc, idx)
        if cid not in chunks:
            chunks[cid] = {"document": doc, "index": idx, "segments": []}
        chunks[cid]["segments"].append(seg)
    # keep segments in time order inside each chunk
    for ch in chunks.values():
        ch["segments"].sort(key=lambda s: (float(s["beg"]), float(s["end"])))
    return chunks


def read_order(path):
    order = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if "chunk_id" in header:
            col = header.index("chunk_id")
        else:  # no header: first line is data, first column is chunk_id
            col = 0
            order.append(header[0])
        for line in f:
            line = line.rstrip("\n")
            if line:
                order.append(line.split("\t")[col])
    return order


def write_order(path, ordered_ids, chunks):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(TSV_HEADER) + "\n")
        for i, cid in enumerate(ordered_ids):
            ch = chunks[cid]
            segs = ch["segments"]
            beg = min(float(s["beg"]) for s in segs)
            end = max(float(s["end"]) for s in segs)
            f.write("\t".join(map(str, [
                i, cid, ch["document"], ch["index"],
                f"{beg:.3f}", f"{end:.3f}", len(segs),
            ])) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input .jsonl ('-' for stdin)")
    ap.add_argument("-o", "--output", required=True, help="output .jsonl ('-' for stdout)")
    ap.add_argument("--chunk-order", required=True,
                    help="TSV with chunk order: read if it exists, otherwise written")
    ap.add_argument("--shuffle", action="store_true",
                    help="always shuffle and (over)write --chunk-order")
    ap.add_argument("--strict", action="store_true",
                    help="error (instead of warning) if the TSV lists chunks absent from input")
    ap.add_argument("--chunk-minutes", type=float, default=20.0,
                    help="chunk length in minutes (default 20)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for shuffling")
    args = ap.parse_args()

    chunks = build_chunks(read_jsonl(args.input), args.chunk_minutes * 60.0)
    if not chunks:
        sys.exit("No segments in input.")

    if args.shuffle or not os.path.exists(args.chunk_order):
        ordered = list(chunks.keys())
        random.Random(args.seed).shuffle(ordered)
        write_order(args.chunk_order, ordered, chunks)
        print(f"Shuffled {len(ordered)} chunks, order written to {args.chunk_order}",
              file=sys.stderr)
    else:
        ordered = read_order(args.chunk_order)
        dup = len(ordered) - len(set(ordered))
        if dup:
            sys.exit(f"{dup} duplicate chunk id(s) in {args.chunk_order}")
        unknown = [c for c in ordered if c not in chunks]
        if unknown:
            msg = (f"{len(unknown)} chunk(s) in {args.chunk_order} have no segments in input, "
                   f"e.g. {unknown[:3]}")
            if args.strict:
                sys.exit(msg)
            print(f"WARNING: {msg} -- skipped", file=sys.stderr)
            ordered = [c for c in ordered if c in chunks]
        missing = [c for c in chunks if c not in set(ordered)]
        if missing:
            print(f"WARNING: {len(missing)} chunk(s) not in {args.chunk_order} are dropped, "
                  f"e.g. {missing[:3]}", file=sys.stderr)
        print(f"Ordered {len(ordered)} chunks according to {args.chunk_order}",
              file=sys.stderr)

    out = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    n = 0
    with out:
        for pos, cid in enumerate(ordered):
            for seg in chunks[cid]["segments"]:
                seg = dict(seg, chunk_id=cid, chunk_order=pos)
                out.write(json.dumps(seg, ensure_ascii=False) + "\n")
                n += 1
    print(f"Wrote {n} segments to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
