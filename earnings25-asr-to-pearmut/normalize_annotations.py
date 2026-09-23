#!/usr/bin/env python3
"""Convert harm_annotation_eng_asr/_all_clips.jsonl into the shared NMEE annotations.jsonl format.

The Earnings25 ASR annotations already follow the agreed schema
(see iwslt+claude-to-pearmut/README.md) except for three things:

  1. no `segment_filename` -- the default guessed by make_pearmut_campaign.py is built from
     Path(filename).stem, which here is the *recording* id (423057182), while the audio and the
     Canary outputs are keyed by *clip* id (423057182_1193.80_1790.84);
  2. `confidence` is "confident"/"borderline", the shared vocabulary is "harmful"/"borderline";
  3. `harm_types` uses the short labels ("Derailing/contresens"), the protocol enum uses the long
     ones ("Derailing or contresens").

With --canary-dir it also adds `cut_start_in_clip` / `cut_end_in_clip`, the span of audio the clip
has to cover. These are normally identical to `seg_start_in_clip` / `seg_end_in_clip`, but for 31
of the 1,905 records the stored `asr` is a character-truncated window running past the end of its
Canary segment and into the next one(s); cutting on the segment boundary would show the annotator
text whose audio is missing, so the clip is extended to the end of the last segment it touches.
The documented `seg_*_in_clip` fields are left untouched.

Everything else is passed through untouched, including the Earnings25-specific extras
(clip_id, company, industry, clip_wer, severity, harm_types_code, intended_*, span_orig_*).

Usage:
    python normalize_annotations.py ../harm_annotation_eng_asr/_all_clips.jsonl -o annotations.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

# short label used in _all_clips.jsonl -> protocol enum in the shared schema
HARM_TYPES = {
    "False attribution": "False attribution",
    "Offensive": "Offensive",
    "Embarrassing/laughable": "Embarrassing or laughable",
    "Derailing/contresens": "Derailing or contresens",
    "Safety/health/legal risk": "Safety, health or legal risk",
    "Other": "Other",
}

CONFIDENCE = {"confident": "harmful", "borderline": "borderline"}


def norm(s):
    return " ".join(s.split())


def cut_bounds(rec, segs):
    """(start, end) in clip time covering every word of rec["asr"].

    `asr` is a prefix of the concatenation of the Canary segments starting at rec["segment"],
    truncated at a fixed character budget, so walk forward until the concatenation is long enough
    and take the end of the last segment consumed.
    """
    k, want = rec["segment"], len(norm(rec["asr"]))
    if k >= len(segs):
        return rec["seg_start_in_clip"], rec["seg_end_in_clip"], False
    acc, j = "", k
    while j < len(segs):
        acc = norm(f"{acc} {segs[j]['segment']}")
        if len(acc) >= want:
            break
        j += 1
    j = min(j, len(segs) - 1)
    return segs[k]["start"], segs[j]["end"], j != k


def segment_filename(rec, ext):
    """./<clip_id>/<clip_id>.<NNNN>.<ext> -- the layout cut_clips.py writes."""
    clip, seg = rec["clip_id"], rec["segment"]
    return f"./{clip}/{clip}.{seg:04d}.{ext}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="_all_clips.jsonl")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--ext", default="mp3", help="extension of the cut clips (default: mp3)")
    ap.add_argument("--canary-dir", help="dir with <clip_id>.en.jsonl; enables the cut_*_in_clip fields")
    args = ap.parse_args()

    n_rec = n_tgt = n_extended = 0
    unknown = set()
    segs_cache = {}
    with open(args.annotations, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rec += 1
            rec["segment_filename"] = segment_filename(rec, args.ext)

            start, end = rec["seg_start_in_clip"], rec["seg_end_in_clip"]
            if args.canary_dir:
                cid = rec["clip_id"]
                if cid not in segs_cache:
                    p = Path(args.canary_dir) / f"{cid}.en.jsonl"
                    segs_cache[cid] = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
                start, end, extended = cut_bounds(rec, segs_cache[cid])
                n_extended += extended
            rec["cut_start_in_clip"], rec["cut_end_in_clip"] = start, end

            for t in rec["targets"]:
                n_tgt += 1
                unknown |= {h for h in t["harm_types"] if h not in HARM_TYPES}
                t["harm_types"] = [HARM_TYPES.get(h, h) for h in t["harm_types"]]
                t["confidence"] = CONFIDENCE.get(t["confidence"], t["confidence"])
                t["borderline"] = t["confidence"] == "borderline"
                # the span offsets are the contract of the whole pipeline; fail loudly, not silently
                if t["text"][t["span_start"]:t["span_end"]] != t["span"]:
                    sys.exit(f"span offsets do not match text in {rec['clip_id']}#{rec['segment']}")
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if unknown:
        print(f"WARNING: harm types not in the protocol enum, passed through: {sorted(unknown)}", file=sys.stderr)
    if args.canary_dir:
        print(f"{n_extended} clips extended past their segment to cover the stored ASR text", file=sys.stderr)
    print(f"{n_rec} segments, {n_tgt} errors -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
