#!/usr/bin/env python3
"""Fill in `filename` and `segment` in annotations.base.jsonl.

The annotations were made on the concatenated file csen_dev_en.txt (one segment per line), so each
record has `concat_line` (1-based). This script finds, for every <doc>.en.jsonl, the contiguous block
of csen_dev_en.txt lines that equals the document's segments, and maps each line to
(filename, 0-based segment index in the jsonl).

With --ref-yaml and --ref-text (the reference segmentation and the gold transcript, aligned line by
line), each annotation also gets the reference segments that overlap it in time:
`ref_lines` (1-based lines in the gold file), `ref_segments` (0-based within the document) and `ref_text`.

Usage:
    python resolve_annotations.py annotations.base.jsonl csen_dev_en.txt \
        --translations-dir ~/work/uedin/mtm26/nmee/outputs/iwslt26-cs-dev \
        --ref-yaml iwslt26-cs-dev.yaml --ref-text iwslt26-cs-dev.cs \
        -o annotations.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path


def load_refs(yaml_path, text_path):
    """Return {doc_stem: [(global_line_1based, local_idx, start, end, text), ...]}."""
    texts = open(text_path, encoding="utf-8").read().rstrip("\n").split("\n")
    try:
        import yaml
        entries = yaml.safe_load(open(yaml_path, encoding="utf-8"))
    except ImportError:  # the file uses one flow mapping per line
        entries = []
        for l in open(yaml_path, encoding="utf-8"):
            if l.strip():
                d = dict(re.findall(r"(\w+):\s*([^,}]+)", l))
                entries.append({"duration": float(d["duration"]), "offset": float(d["offset"]),
                                "wav": d["wav"].strip()})
    if len(entries) != len(texts):
        sys.exit(f"{yaml_path} has {len(entries)} segments but {text_path} has {len(texts)} lines")
    refs = {}
    for i, (e, t) in enumerate(zip(entries, texts)):
        stem = Path(e["wav"]).stem
        lst = refs.setdefault(stem, [])
        lst.append((i + 1, len(lst), e["offset"], e["offset"] + e["duration"], t))
    return refs


def overlapping(refs, start, end, min_overlap):
    out = [r for r in refs if min(end, r[3]) - max(start, r[2]) > min_overlap]
    if not out:  # fall back to the nearest reference segment
        out = [min(refs, key=lambda r: abs((r[2] + r[3]) / 2 - (start + end) / 2))]
    return out


def norm(s):
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations")
    ap.add_argument("concat", help="csen_dev_en.txt, i.e. all segments concatenated, one per line")
    ap.add_argument("--translations-dir", required=True)
    ap.add_argument("--suffix", default=".en.jsonl")
    ap.add_argument("--audio-ext", default=".wav", help="extension used for `filename` (default: .wav)")
    ap.add_argument("--ref-yaml", help="reference segmentation (iwslt26-cs-dev.yaml)")
    ap.add_argument("--ref-text", help="gold transcript aligned with --ref-yaml (iwslt26-cs-dev.cs)")
    ap.add_argument("--min-overlap", type=float, default=0.1,
                    help="seconds of overlap needed to attach a reference segment (default 0.1)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    refs = load_refs(args.ref_yaml, args.ref_text) if args.ref_yaml and args.ref_text else None

    lines = [norm(l) for l in open(args.concat, encoding="utf-8").read().split("\n")]
    line2loc = {}  # 1-based concat line -> (stem, seg_idx)
    total = 0
    for jf in sorted(Path(args.translations_dir).glob(f"*{args.suffix}")):
        stem = jf.name[: -len(args.suffix)]
        recs = [json.loads(l) for l in open(jf, encoding="utf-8") if l.strip()]
        segs = [norm(r["segment"]) for r in recs]
        if not segs:
            continue
        total += len(segs)
        hits = [i for i in range(len(lines) - len(segs) + 1)
                if lines[i] == segs[0] and lines[i:i + len(segs)] == segs]
        if len(hits) != 1:
            print(f"WARNING: {jf.name}: {len(hits)} block matches in {args.concat}", file=sys.stderr)
            continue
        for k in range(len(segs)):
            line2loc[hits[0] + k + 1] = (stem, k, recs[k].get("start"), recs[k].get("end"))

    n_nonempty = sum(1 for l in lines if l)
    print(f"{total} segments in jsonl files, {n_nonempty} non-empty lines in concat, "
          f"{len(line2loc)} lines mapped", file=sys.stderr)

    ok = missing = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for l in open(args.annotations, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            loc = line2loc.get(r["concat_line"])
            if loc is None:
                missing += 1
                print(f"UNRESOLVED line {r['concat_line']}: {r['span']!r}", file=sys.stderr)
            else:
                stem, k, start, end = loc
                r["filename"] = stem + args.audio_ext
                r["segment"] = k
                r["start"], r["end"] = start, end
                if refs is not None:
                    if stem not in refs:
                        print(f"WARNING: {stem} not in {args.ref_yaml}", file=sys.stderr)
                    else:
                        ov = overlapping(refs[stem], start, end, args.min_overlap)
                        r["ref_lines"] = [x[0] for x in ov]
                        r["ref_segments"] = [x[1] for x in ov]
                        r["ref_text"] = " ".join(x[4] for x in ov)
                ok += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"resolved {ok}, unresolved {missing} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
