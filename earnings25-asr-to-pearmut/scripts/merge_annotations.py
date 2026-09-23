#!/usr/bin/env python3
"""Merge the two Earnings25 error-annotation sets onto one common grid.

The two sets cover the same 290 Earnings25 clips but disagree about almost everything else:

  harm_annotation_eng_asr/            dspy-search-for-errors/35-merged-annotations.jsonl
  -----------------------------------  -------------------------------------------------
  unit: one Canary *en* segment        unit: a gold-transcript word range, mapped to time
  1 system (ASR+canary), English only  5 languages (en + de/cs/pl/sk MT)
  exact character offsets into `asr`   offsets into the model's own quoted rendering
  harm taxonomy + high/medium/low      harm_types always ["Other"]; error class in the text
  one annotator (claude-opus-5)        three (GLM, Kimi, gpt-oss-120B) voting independently

This script puts both on the **Canary English segment grid** -- the unit the first set already
uses, and the unit the audio clips are cut to. Each divergence record is attached to the Canary
segment it overlaps most; 91% of them touch exactly one.

  - English. A divergence record's `automatic_texts["en"]` is a window of the Canary transcript,
    so its spans are re-located inside the full Canary segment text by string search and become
    extra spans on the *same* target as the first set's. This is the part where the two sets
    genuinely overlap and can be compared.
  - de/cs/pl/sk. There is no per-Canary-segment MT text to merge into: the per-language Canary
    outputs are independently segmented (88/84/89/90/87 segments for the same clip), so they
    cannot be indexed by the English segment number. The divergence record's own time-aligned
    `automatic_texts[lang]` is used as the target text instead, taking the largest-overlapping
    window when several map to one segment and re-locating the other windows' spans into it.

Nothing is dropped: a span that cannot be re-located keeps its explanation and simply is not
pre-highlighted. The three divergence models are collapsed per (language, span), and how many of
them reported it is kept in `num_models_reporting`.

Usage:
    python scripts/merge_annotations.py \
        --asr-harm annotations/annotations.jsonl \
        --divergencies work/35-merged-annotations.jsonl \
        --canary-dir ../../results/canary/en \
        --index ../harm_annotation_eng_asr/_index.json \
        --min-models 2 -o annotations/merged_annotations.jsonl
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LANGS = ["en", "de", "cs", "pl", "sk"]
SYSTEM = {"en": "ASR+canary", "de": "canary-st (de)", "cs": "canary-st (cs)",
          "pl": "canary-st (pl)", "sk": "canary-st (sk)"}


def norm(s):
    return " ".join(s.split())


def load_canary(canary_dir, clip_id, cache):
    if clip_id not in cache:
        p = Path(canary_dir) / f"{clip_id}.en.jsonl"
        cache[clip_id] = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if p.exists() else []
    return cache[clip_id]


def best_segment(segs, start, end):
    """Index of the Canary segment overlapping [start, end] most, and that overlap."""
    best, best_ov = None, 0.0
    for i, s in enumerate(segs):
        ov = min(end, s["end"]) - max(start, s["start"])
        if ov > best_ov:
            best, best_ov = i, ov
    return best, best_ov


def relocate(span, text, hint_text=None):
    """(start, end, matched) of `span` inside `text`, or None.

    `hint_text` is the window the span was measured against; when it sits inside `text` the
    search is anchored there, which avoids matching an earlier identical word in the segment.
    The match is returned as well, because the case-insensitive fallback can match a differently
    cased string -- storing `span` unchanged there would break `text[start:end] == span`.
    """
    if not span:
        return None
    if hint_text:
        base = text.find(hint_text)
        if base >= 0:
            i = hint_text.find(span)
            if i >= 0:
                return base + i, base + i + len(span), span
    i = text.find(span)
    if i >= 0:
        return i, i + len(span), span
    lo = text.lower().find(span.lower())
    return (lo, lo + len(span), text[lo:lo + len(span)]) if lo >= 0 else None


def divergence_targets(rec, lang, texts_for_lang, seg_text):
    """Collapse one divergence record's targets for one language into unique spans."""
    by_span = {}
    for t in rec["targets"]:
        if t["tgt_lan"] != lang:
            continue
        key = (t["span_start"], t["span_end"], t["span"])
        e = by_span.setdefault(key, {"models": set(), "explanations": [], "t": t})
        if t.get("annotator_model"):
            e["models"].add(t["annotator_model"])
        if t["explanation"] not in e["explanations"]:
            e["explanations"].append(t["explanation"])
    return by_span


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr-harm", required=True, help="annotations.jsonl from normalize_annotations.py")
    ap.add_argument("--divergencies", required=True, help="35-merged-annotations.jsonl")
    ap.add_argument("--canary-dir", required=True, help="dir with <clip_id>.en.jsonl")
    ap.add_argument("--index", help="harm_annotation_eng_asr/_index.json, for company names")
    ap.add_argument("--min-models", type=int, default=1,
                    help="keep divergence records reported by at least this many models (default: 1)")
    ap.add_argument("--langs", default=",".join(LANGS), help=f"languages to keep (default: {','.join(LANGS)})")
    ap.add_argument("--ext", default="mp3")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    langs = [l for l in args.langs.split(",") if l]
    companies = {}
    if args.index:
        companies = {c["clip_id"]: c.get("company") for c in json.load(open(args.index, encoding="utf-8"))}

    cache = {}
    merged = {}  # (clip_id, segment) -> record

    # ---- 1. the ASR-harm set: already on the grid, taken as-is --------------------------------
    n_harm = 0
    for line in open(args.asr_harm, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for t in r["targets"]:
            t.setdefault("source", "asr-harm")
            t["system"] = SYSTEM["en"]
            n_harm += 1
        r["targets"] = [t for t in r["targets"] if t["tgt_lan"] in langs]
        merged[(r["clip_id"], r["segment"])] = r

    # ---- 2. the divergence set: attached to the segment it overlaps most -----------------------
    by_segment = defaultdict(list)
    n_div_rec = 0
    for line in open(args.divergencies, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d["num_models_reporting"] < args.min_models:
            continue
        segs = load_canary(args.canary_dir, d["id"], cache)
        if not segs:
            continue
        k, ov = best_segment(segs, d["orig_start"], d["orig_end"])
        if k is None:
            continue
        by_segment[(d["id"], k)].append((ov, d))
        n_div_rec += 1

    n_div_tgt = n_reloc = n_unreloc = 0
    for (clip_id, k), windows in by_segment.items():
        windows.sort(key=lambda x: -x[0])  # largest overlap first: it provides the MT text
        segs = load_canary(args.canary_dir, clip_id, cache)
        seg = segs[k]
        rec = merged.get((clip_id, k))
        if rec is None:
            rec = new_record(clip_id, k, seg, windows[0][1], companies, args.ext)
            merged[(clip_id, k)] = rec

        # the audio has to cover every window attached to this segment
        lo = min([rec["cut_start_in_clip"]] + [d["orig_start"] for _, d in windows])
        hi = max([rec["cut_end_in_clip"]] + [d["orig_end"] for _, d in windows])
        rec["cut_start_in_clip"], rec["cut_end_in_clip"] = lo, hi

        for lang in langs:
            # the target text: English is the Canary segment, MT is the widest window's rendering
            if lang == "en":
                # not seg["segment"]: for the 32 overflow records the stored ASR is a longer
                # window, and the asr-harm offsets are measured against that
                text = rec["asr"]
            else:
                text = next((d["automatic_texts"].get(lang) for _, d in windows
                             if d["automatic_texts"].get(lang)), None)
                if not text:
                    continue

            for _, d in windows:
                hint = d["automatic_texts"].get(lang)
                for (s0, s1, span), e in divergence_targets(d, lang, None, text).items():
                    n_div_tgt += 1
                    t = dict(e["t"])
                    whole = span == hint  # "the whole rendering is wrong" -- nothing to pinpoint
                    off = None if whole else relocate(span, text, hint)
                    if whole:
                        pass
                    elif off:
                        n_reloc += 1
                    else:
                        n_unreloc += 1
                    t.update({
                        "system": SYSTEM[lang], "tgt_lan": lang, "text": text,
                        "span": off[2] if (off and not whole) else "",
                        "span_start": off[0] if (off and not whole) else None,
                        "span_end": off[1] if (off and not whole) else None,
                        "span_whole_output": whole,
                        "explanation": " | ".join(e["explanations"]),
                        "error_class": error_class(e["explanations"]),
                        "num_models_reporting": len(e["models"]),
                        "models_reporting": sorted(e["models"]),
                        "source": "dspy-divergencies",
                        "annotator": "GLM/Kimi/gpt-oss-120B (LLM vote), transcript-only",
                        "divergence_window": [d["orig_start"], d["orig_end"]],
                        "golden_word_range": d.get("golden_word_range"),
                    })
                    for gone in ("annotator_model", "span_matched", "severity", "reference"):
                        t.pop(gone, None)
                    rec["targets"].append(t)

    out = [merged[key] for key in sorted(merged) if merged[key]["targets"]]
    with open(args.output, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tgt = sum(len(r["targets"]) for r in out)
    both = sum(1 for r in out if {t["source"] for t in r["targets"]} == {"asr-harm", "dspy-divergencies"})
    print(f"asr-harm targets in:      {n_harm}", file=sys.stderr)
    print(f"divergence records kept:  {n_div_rec} (>= {args.min_models} model(s))", file=sys.stderr)
    print(f"divergence spans:         {n_div_tgt} -> {n_reloc} re-located, {n_unreloc} not found "
          f"(kept, not pre-highlighted)", file=sys.stderr)
    print(f"{len(out)} segments, {n_tgt} targets, {both} segments flagged by both sets "
          f"-> {args.output}", file=sys.stderr)


def error_class(explanations):
    """30-merge-annotations.py formats each explanation as '<error class>: <text>'."""
    classes = []
    for e in explanations:
        m = re.match(r"([a-z][a-z /-]{2,40}?):\s", e)
        if m and m.group(1) not in classes:
            classes.append(m.group(1))
    return ", ".join(classes) or None


def new_record(clip_id, k, seg, d, companies, ext):
    """A segment the ASR-harm set never flagged, so it has to be built from the Canary grid."""
    recording, cstart, cend = clip_id.split("_")
    cstart, cend = float(cstart), float(cend)
    return {
        "doc_id": f"earnings-25/{recording}",
        "filename": f"{recording}.mp3",
        "clip_id": clip_id,
        "clip_filename": f"audio/{clip_id}.mp3",
        "clip_orig_start": cstart,
        "clip_orig_end": cend,
        "company": companies.get(clip_id),
        "segment": k,
        "orig_start": cstart + seg["start"],
        "orig_end": cstart + seg["end"],
        "seg_start_in_clip": seg["start"],
        "seg_end_in_clip": seg["end"],
        "cut_start_in_clip": seg["start"],
        "cut_end_in_clip": seg["end"],
        "segment_filename": f"./{clip_id}/{clip_id}.{k:04d}.{ext}",
        "asr": seg["segment"],
        "gold_transcript": d.get("gold_transcript"),
        "targets": [],
    }


if __name__ == "__main__":
    main()
