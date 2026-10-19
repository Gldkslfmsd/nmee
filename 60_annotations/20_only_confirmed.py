#!/usr/bin/env python3
"""Print only the segments that human annotator(s) confirmed as containing a harmful span.

    python3 only_confirmed.py gold/Kimi_cs.gold.jsonl gold/Kimi_fr.gold.jsonl > confirmed.jsonl
    python3 only_confirmed.py gold/*.gold.jsonl > confirmed.jsonl
    python3 only_confirmed.py gold/Kimi_cs.*.jsonl --any > confirmed.jsonl
    python3 only_confirmed.py gold/*.gold.jsonl --no-asr --lan cs de > confirmed.jsonl

Input: one or more outputs of pearmut_to_jsonl.py (JSONL, one segment per line).

Combining several files: records are matched by the audio file name. The targets of the same segment
from different files are put together in one record; a system that is in more than one file is merged
(human spans of all annotators, AI-suggested spans of all files).

A target is confirmed harmful when
  default: every annotator who saw it marked an error span in it (the same rule as in pearmut_to_jsonl.py)
  --any:   at least one annotator who saw it marked an error span in it
A segment is printed if at least one of its targets is confirmed harmful; only the confirmed targets
are kept in it. --no-asr and --lan remove targets before this is decided, so a segment whose only
confirmed target is removed is not printed.

Output format (JSONL on stdout, or -o FILE), one segment per line:
  document, dataset, src_language, audio, beg, end, asr, asr_system, gold_transcript,
  targets: [{tgt_lan, system, text,
             harm_spans: [{span, span_start, span_end}],      # human-confirmed, end exclusive
             ai_suggested_spans: [{span, span_start, span_end, intended, harm_types, harmfulness,
                                   explanation, error_source}]}]
Statistics go to stderr.
"""
import argparse
import json
import os
import sys
from collections import Counter, OrderedDict

NO_USER = "_combined"  # used for records without "gold_by_annotator"

SEGMENT_FIELDS = ("document", "dataset", "src_language", "audio", "beg", "end",
                  "asr", "asr_system", "gold_transcript")
AI_SPAN_FIELDS = ("span", "span_start", "span_end", "intended", "harm_types", "harmfulness",
                  "explanation", "error_source")


def shout(msg):
    print(msg, file=sys.stderr)


def audio_name(rec):
    return os.path.basename(rec.get("audio") or "") or f"{rec.get('document')}_{rec.get('beg')}_{rec.get('end')}"


def load(paths, stats):
    """{segment key: {"rec": first record, "targets": OrderedDict(system -> {"target", "ai", "users"})}}"""
    segments = OrderedDict()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                stats["input records"] += 1
                if "gold_annotations" not in rec:
                    shout(f"WARNING: {path}:{n}: no gold_annotations (not an output of pearmut_to_jsonl.py?), skipped")
                    stats["records without gold_annotations (skipped)"] += 1
                    continue
                key = audio_name(rec)
                seg = segments.get(key)
                if seg is None:
                    seg = segments[key] = {"rec": rec, "targets": OrderedDict()}
                else:
                    stats["records merged into an earlier segment"] += 1
                targets = rec.get("targets") or []
                by_ann = rec.get("gold_by_annotator")
                if by_ann is None:  # no per-annotator info: take the combined verdict as one annotator
                    by_ann = {NO_USER: {"gold_annotations": rec["gold_annotations"],
                                        "gold_annotation_details": rec.get("gold_annotation_details")
                                        or [None] * len(targets),
                                        "gold_spans": rec.get("gold_spans") or [None] * len(targets)}}
                for k, t in enumerate(targets):
                    entry = seg["targets"].setdefault(t.get("system"),
                                                      {"target": t, "ai": [], "users": {}})
                    if t.get("span") is not None:  # AI-flagged span; the same one may come from several files
                        ai = {f: t.get(f) for f in AI_SPAN_FIELDS}
                        sig = (ai["span_start"], ai["span_end"], ai["intended"])
                        if all(sig != (a["span_start"], a["span_end"], a["intended"]) for a in entry["ai"]):
                            entry["ai"].append(ai)
                    for user, a in by_ann.items():
                        verdict = a["gold_annotations"][k]
                        if verdict is None:
                            continue
                        entry["users"][user] = {"gold": verdict, "spans": a["gold_spans"][k] or []}
    return segments


def merge_spans(span_lists):
    """Union of the human spans of several annotators; overlapping spans become one."""
    spans = sorted((s for lst in span_lists for s in lst), key=lambda s: (s["start"], s["end"]))
    merged = []
    for s in spans:
        if merged and s["start"] < merged[-1]["end"]:
            if s["end"] > merged[-1]["end"]:
                merged[-1] = {"start": merged[-1]["start"], "end": s["end"],
                              "text": merged[-1]["text"] + s["text"][merged[-1]["end"] - s["start"]:]}
        else:
            merged.append(dict(s))
    return merged


def combine(users, any_mode):
    """(confirmed harmful?, merged human spans) for one target; None if no annotator saw it."""
    if not users:
        return None, []
    golds = [u["gold"] for u in users.values()]
    gold = any(golds) if any_mode else all(golds)
    if not gold:
        return False, []
    return True, merge_spans([u["spans"] for u in users.values() if u["gold"]])


def selected(system, entry, asr_system, no_asr, langs):
    """Is this target wanted? The ASR is decided by --no-asr only; translations by --lan."""
    if system == asr_system:
        return not no_asr
    if langs is None:
        return True
    return (entry["target"].get("tgt_lan") or "").lower() in langs


def build(seg, any_mode, no_asr=False, langs=None):
    """The output record with the confirmed targets only, or None if no target is confirmed."""
    asr_system = seg["rec"].get("asr_system")
    wanted = {s: e for s, e in seg["targets"].items() if selected(s, e, asr_system, no_asr, langs)}
    verdicts = {s: combine(e["users"], any_mode) for s, e in wanted.items()}
    if not any(v[0] for v in verdicts.values()):
        return None
    out = {f: seg["rec"].get(f) for f in SEGMENT_FIELDS}
    out["targets"] = []
    for system, entry in wanted.items():
        confirmed, spans = verdicts[system]
        if not confirmed:
            continue
        t = entry["target"]
        out["targets"].append({
            "tgt_lan": t.get("tgt_lan"),
            "system": system,
            "text": t.get("text"),
            "harm_spans": [{"span": s["text"], "span_start": s["start"], "span_end": s["end"]}
                           for s in spans],
            "ai_suggested_spans": entry["ai"],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of pearmut_to_jsonl.py")
    ap.add_argument("-o", "--output", help="output file (default: stdout)")
    ap.add_argument("--any", action="store_true", dest="any_mode",
                    help="a target is confirmed if at least one annotator marked it (default: all who saw it)")
    ap.add_argument("--no-asr", action="store_true",
                    help="leave out the ASR target (the system in asr_system)")
    ap.add_argument("--lan", nargs="+", metavar="LAN",
                    help="target languages of the translations to include, e.g. --lan cs fr (default: all). "
                         "The ASR target is not affected, use --no-asr for it")
    args = ap.parse_args()
    langs = {l.lower() for l in args.lan} if args.lan else None

    stats = Counter()
    segments = load(args.files, stats)
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    try:
        for seg in segments.values():
            stats["segments"] += 1
            rec = build(seg, args.any_mode, args.no_asr, langs)
            if rec is None:
                continue
            stats["segments confirmed harmful (printed)"] += 1
            stats["confirmed targets (printed)"] += len(rec["targets"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if args.output:
            out.close()
    for k, v in stats.items():
        shout(f"  {k}: {v}")


if __name__ == "__main__":
    main()
