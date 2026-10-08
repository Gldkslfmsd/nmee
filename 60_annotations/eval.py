#!/usr/bin/env python3
"""Evaluate the LLM preselection against human annotations: the output of pearmut_to_jsonl.py.

    python3 eval.py gold/Kimi_cs.jsonl [more files ...] [--input-segments aligned/all.jsonl]

Every item of "targets" is one decision, paired with its entry in "gold_annotations":
  predicted positive  the LLM flagged a span ("span" is not null)
  predicted negative  a target without a span (the LLM flagged nothing in this system and segment)
  gold                true / false from the annotator; null entries are skipped
                      "gold no" is split by "gold_annotation_details" into no_harm (the annotator chose
                      [no harm]) and undecidable (the annotator chose [undecidable]); both count as gold no

For each system (the ASR marked as such), for all translations together, and overall, it prints the
confusion matrix and accuracy, precision, recall and F1, and then a summary of the audio duration of the
evaluated segments (end - beg; if missing, read from the wav file when it exists), in total and for the
segments with at least one human-confirmed error.

--input-segments: the aligned JSONL of all segments (input of 40_find_harmful_errors.py; "document",
"audio", "beg", "end", ...). Then it also prints how many segments and how much audio the original
documents have, which part of it was annotated, and harm rates: human-confirmed errors (and LLM flags)
per hour of annotated audio, per hour of the original documents and per 100 original segments.
The rates over the original documents count only the annotated segments, so they are lower bounds unless
every LLM-preselected segment of these documents was annotated.

Caveats: the segments are the ones the LLM preselected (only those were annotated), so recall is recall
within the preselected segments, not over the whole data. Gold is a verdict on the whole output of a
system: true if the annotator marked a harmful error anywhere in it, not necessarily at the span the LLM
flagged (see pearmut_to_jsonl.py).
"""
import argparse
import json
import os
import statistics
import sys
import wave
from collections import Counter, OrderedDict


def load(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records += [json.loads(l) for l in f if l.strip()]
    return records


def count(records):
    """{group: Counter(tp, fp, fn, tn)}, Counter of skipped items, {system: is_asr}."""
    groups, skipped, is_asr = OrderedDict(), Counter(), {}
    for rec in records:
        targets, gold = rec.get("targets", []), rec.get("gold_annotations")
        if gold is None or len(gold) != len(targets):
            skipped["records without matching gold_annotations"] += 1
            continue
        details = rec.get("gold_annotation_details")
        if details is None or len(details) != len(targets):
            details = [None] * len(targets)  # older files without details: nothing is undecidable
        for t, g, d in zip(targets, gold, details):
            if g is None:
                skipped["targets with null gold"] += 1
                continue
            system = t.get("system")
            asr = system == rec.get("asr_system")
            is_asr[system] = is_asr.get(system, False) or asr
            pred = t.get("span") is not None
            cell = "tp" if pred and g else "fp" if pred else "fn" if g else "tn"
            kind = None if g else "un" if d == "undecidable" else "nh" if d == "no_harm" else None
            for group in (system, "all translations" if not asr else None, "overall"):
                if group:
                    groups.setdefault(group, Counter())[cell] += 1
                    if kind:  # split of the gold-no cells (fp, tn) by the annotator's token
                        groups[group][f"{cell}_{kind}"] += 1
    return groups, skipped, is_asr


def ratio(a, b):
    return a / b if b else None


def fmt(v):
    return "    -" if v is None else f"{v:.3f}"


def metrics(c):
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    p, r = ratio(c["tp"], c["tp"] + c["fp"]), ratio(c["tp"], c["tp"] + c["fn"])
    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None
    return n, ratio(c["tp"] + c["tn"], n), p, r, f1


def fmt_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def segment_duration(rec):
    """Duration in seconds from beg/end, else from the wav file, else None."""
    beg, end = rec.get("beg"), rec.get("end")
    if isinstance(beg, (int, float)) and isinstance(end, (int, float)) and end >= beg:
        return end - beg
    path = rec.get("audio")
    if path and os.path.exists(path):
        try:
            with wave.open(path) as w:
                return w.getnframes() / w.getframerate()
        except (wave.Error, OSError, ZeroDivisionError):
            return None
    return None


def segment_name(rec):
    """Segment identity shared by all formats: the audio file name (else document, beg, end)."""
    return os.path.basename(rec.get("audio") or "") or f"{rec.get('document')}\t{rec.get('beg')}\t{rec.get('end')}"


def load_segments(paths):
    """{segment name: {"document", "duration"}} of the aligned input JSONL; and the number of duplicates."""
    segments, duplicates = {}, 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = segment_name(rec)
                if name in segments:
                    duplicates += 1
                    continue
                segments[name] = {"document": rec.get("document"), "duration": segment_duration(rec)}
    return segments, duplicates


def hours(seconds):
    return seconds / 3600 if seconds else 0


def per(count, denominator, scale=1.0):
    return f"{count / denominator * scale:.2f}" if denominator else "-"


def original_summary(records, groups, is_asr, order, segments, duplicates, out=sys.stdout):
    """Size of the original documents, the annotated part of them, and harm rates."""
    eval_names = {segment_name(r) for r in records}
    found = eval_names & set(segments)
    docs = {segments[n]["document"] for n in found}
    doc_segments = [x for x in segments.values() if x["document"] in docs]
    all_dur = sum(x["duration"] or 0 for x in segments.values())
    doc_dur = sum(x["duration"] or 0 for x in doc_segments)
    ann_dur = sum(segments[n]["duration"] or 0 for n in found)
    n_all_docs = len({x["document"] for x in segments.values()})

    print("\noriginal documents (--input-segments)", file=out)
    print(f"  input                     {len(segments):>7} segments, {n_all_docs:>4} documents, "
          f"audio {fmt_duration(all_dur)}", file=out)
    print(f"  documents in evaluation   {len(doc_segments):>7} segments, {len(docs):>4} documents, "
          f"audio {fmt_duration(doc_dur)}", file=out)
    print(f"  annotated                 {len(found):>7} segments "
          f"({per(len(found), len(doc_segments), 100)}% of these documents), "
          f"audio {fmt_duration(ann_dur)} ({per(ann_dur, doc_dur, 100)}%)", file=out)
    if eval_names - found:
        print(f"  WARNING: {len(eval_names - found)} evaluated segments are not in --input-segments",
              file=out)
    if duplicates:
        print(f"  note: {duplicates} duplicate segments in --input-segments counted once", file=out)

    h_ann, h_doc = hours(ann_dur), hours(doc_dur)
    print(f"\nharm rates  (errors = human-confirmed: gold yes; flags = LLM-flagged spans; "
          f"/h = per hour of audio)", file=out)
    print(f"{'':<24}{'errors':>8}{'/h annot.':>11}{'/h docs':>9}{'/100 seg':>10}"
          f"{'flags':>7}{'flags/h':>9}{'TP/h':>7}", file=out)
    for group in order:
        if group not in groups:
            continue
        c = groups[group]
        errors, flags = c["tp"] + c["fn"], c["tp"] + c["fp"]
        label = f"{group} (ASR)" if is_asr.get(group) else group
        print(f"{label:<24}{errors:>8}{per(errors, h_ann):>11}{per(errors, h_doc):>9}"
              f"{per(errors, len(doc_segments), 100):>10}{flags:>7}{per(flags, h_doc):>9}"
              f"{per(c['tp'], h_doc):>7}", file=out)
    with_error = sum(1 for r in records if segment_name(r) in found
                     and any(g is True for g in r.get("gold_annotations") or []))
    print(f"{'segments with an error':<24}{with_error:>8}{per(with_error, h_ann):>11}"
          f"{per(with_error, h_doc):>9}{per(with_error, len(doc_segments), 100):>10}", file=out)
    print("  /h docs and /100 seg count only annotated segments: lower bounds unless every "
          "preselected segment of these documents was annotated", file=out)


def audio_summary(records, out=sys.stdout):
    segments = {}  # one entry per segment, even if it is in several input files
    for rec in records:
        key = segment_name(rec)
        has_error = any(g is True for g in rec.get("gold_annotations") or [])
        if key in segments:
            segments[key] = (segments[key][0], segments[key][1] or has_error, segments[key][2])
        else:
            segments[key] = (segment_duration(rec), has_error, rec.get("document"))
    known = [(d, e, doc) for d, e, doc in segments.values() if d is not None]
    unknown = len(segments) - len(known)

    print("\naudio duration", file=out)
    if not known:
        print("  unknown (no beg/end and no readable audio)", file=out)
        return
    for label, rows in (("all segments", known),
                        ("segments with a confirmed error", [x for x in known if x[1]])):
        ds = [d for d, _, _ in rows]
        if not ds:
            print(f"  {label:<34} 0 segments", file=out)
            continue
        docs = len({doc for _, _, doc in rows})
        print(f"  {label:<34}{len(ds):>5} segments, {docs:>3} documents, total {fmt_duration(sum(ds))} "
              f"| per segment: mean {statistics.mean(ds):.1f}s, median {statistics.median(ds):.1f}s, "
              f"min {min(ds):.1f}s, max {max(ds):.1f}s", file=out)
    if unknown:
        print(f"  duration unknown for {unknown} segments (no beg/end and no readable audio)", file=out)


def report(records, title=None, out=sys.stdout, segments=None, duplicates=0):
    """segments: output of load_segments() for the statistics of the original documents, or None."""
    groups, skipped, is_asr = count(records)
    systems = [g for g in groups if g not in ("all translations", "overall")]
    order = ([s for s in systems if is_asr.get(s)] + [s for s in systems if not is_asr.get(s)]
             + (["all translations"] if sum(not is_asr.get(s) for s in systems) > 1 else [])
             + ["overall"])
    print("=" * 72, file=out)
    print(f"EVALUATION{': ' + title if title else ''}  ({len(records)} segments)", file=out)
    print("=" * 72, file=out)
    for group in order:
        if group not in groups:
            continue
        c = groups[group]
        n, acc, p, r, f1 = metrics(c)
        label = f"{group} (ASR)" if is_asr.get(group) else group
        print(f"\n{label}  ({n} targets)", file=out)
        split = any(c[k] for k in ("fp_nh", "fp_un", "tn_nh", "tn_un"))  # files without details: no split
        head = f"  | {'no_harm':>8}{'undecidable':>13}" if split else ""
        flagged_split = f"  | {c['fp_nh']:>8}{c['fp_un']:>13}" if split else ""
        unflagged_split = f"  | {c['tn_nh']:>8}{c['tn_un']:>13}" if split else ""
        print(f"{'':>16}{'gold yes':>10}{'gold no':>10}{head}", file=out)
        print(f"{'flagged':>16}{c['tp']:>10}{c['fp']:>10}{flagged_split}", file=out)
        print(f"{'not flagged':>16}{c['fn']:>10}{c['tn']:>10}{unflagged_split}", file=out)
        print(f"  accuracy {fmt(acc)} | precision {fmt(p)} | recall {fmt(r)} | F1 {fmt(f1)}", file=out)

    print(f"\n{'summary':<24}{'n':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}"
          f"{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}", file=out)
    for group in order:
        if group in groups:
            c = groups[group]
            n, acc, p, r, f1 = metrics(c)
            label = f"{group} (ASR)" if is_asr.get(group) else group
            print(f"{label:<24}{n:>6}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}{c['tn']:>6}"
                  + "".join(f"{fmt(v):>8}" for v in (acc, p, r, f1)), file=out)
    audio_summary(records, out)
    if segments is not None:
        original_summary(records, groups, is_asr, order, segments, duplicates, out)
    for k, v in sorted(skipped.items()):
        print(f"  skipped: {k}: {v}", file=out)
    print("  note: only LLM-preselected segments were annotated; recall is within those segments",
          file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of pearmut_to_jsonl.py (merged)")
    ap.add_argument("--input-segments", nargs="+",
                    help="aligned JSONL of all segments of the original documents (merged)")
    args = ap.parse_args()
    segments, duplicates = load_segments(args.input_segments) if args.input_segments else (None, 0)
    report(load(args.files), title=", ".join(args.files), segments=segments, duplicates=duplicates)


if __name__ == "__main__":
    main()