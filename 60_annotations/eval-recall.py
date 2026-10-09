#!/usr/bin/env python3
"""How much text is inside the harmful spans the annotator marked, and how much of it does the LLM flag?

    python3 eval-recall.py gold/Kimi_cs.jsonl [more files ...] [--input-segments aligned/all.jsonl]

Same input and options as eval.py: the output(s) of pearmut_to_jsonl.py (or intersect.py), and optionally the
aligned JSONL of all segments. The input must have "gold_spans" (the human error spans); files made by an
older pearmut_to_jsonl.py have to be generated again.

The unit is the output of one system (the ASR or a translation) in one segment: one text the annotator saw
and judged. Its gold is as in eval.py (true / false; null is skipped; the same segment and system in
several records is counted once, the first one wins). For each unit:
  human spans   the error spans the annotator marked ("gold_spans")
  LLM spans     the spans the LLM flagged in this system output ("span" of its targets)
Characters are counted with spaces, words are whitespace-separated tokens (for scripts written without
spaces, use the characters). A word is inside a span if they share at least one character. Overlapping spans
are counted once in the characters and words, but every span counts in the number of spans.

For each system, for all translations together and overall, it prints
  - the size of the harmful spans: spans, characters, words, their share of the whole text, and the length
    of one span (mean, median, max),
  - the same for the LLM-flagged spans,
  - the coverage table: recall = how much of the human harmful material is inside an LLM span, precision =
    how much of the LLM-flagged material is inside a human span, for characters, words, spans (a span counts
    if it overlaps at least one span of the other side) and outputs (a harmful output counts if the LLM
    flagged something in it; this is the recall and precision of eval.py), with a 95% Wilson interval for
    spans and outputs.

--input-segments: the aligned JSONL of all segments (as for eval.py). Then it also prints how many segments
and how much audio the input has, how much of it was annotated, harmful spans and words per hour of
annotated audio and per 100 annotated segments, and an estimate of their number in the whole input.

Caveat: the numbers describe the annotated segments. They estimate the whole data only if the annotated
segments are a random sample of it (the estimate for the input assumes this); if the LLM preselected them,
recall is only recall within the preselected segments (see eval.py).
"""
import argparse
import math
import re
import statistics
import sys
from collections import Counter, OrderedDict

import eval as evaluation  # eval.py next to this script

WORD = re.compile(r"\S+")


# ---------------------------------------------------------------- intervals

def place(text, start, end, expected):
    """(start, end) of a span in `text`: the given offsets if they fit, else the first occurrence of the span."""
    if (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)
            and (expected is None or text[start:end] == expected)):
        return start, end
    if expected:
        i = text.find(expected)
        if i >= 0:
            return i, i + len(expected)
    return None


def merge(intervals):
    """Sorted, non-overlapping intervals covering the same characters."""
    out = []
    for s, e in sorted(intervals):
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def length(merged):
    return sum(e - s for s, e in merged)


def intersection(a, b):
    """Number of characters inside both lists of merged intervals."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def touches(interval, merged):
    s, e = interval
    return any(hs < e and s < he for hs, he in merged)


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - width) / denom, (centre + width) / denom


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "-"


def rate(a, b):
    return f"{a / b:.3f}" if b else "    -"


def ci_text(k, n):
    ci = wilson(k, n)
    return "" if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]"


# ---------------------------------------------------------------- units

def build_units(records):
    """One unit per (segment, system): text, kind, human spans, LLM spans. Returns units, skipped, saw_spans."""
    units, skipped, seen, saw_spans = [], Counter(), set(), False
    for rec in records:
        targets, gold = rec.get("targets", []), rec.get("gold_annotations")
        if gold is None or len(gold) != len(targets):
            skipped["records without matching gold_annotations"] += 1
            continue
        details = rec.get("gold_annotation_details")
        if not isinstance(details, list) or len(details) != len(targets):
            details = [None] * len(targets)
        gspans = rec.get("gold_spans")
        if isinstance(gspans, list) and len(gspans) == len(targets):
            saw_spans = True
        else:
            gspans = [None] * len(targets)
        by_system = OrderedDict()
        for item in zip(targets, gold, details, gspans):
            by_system.setdefault(item[0].get("system"), []).append(item)
        name = evaluation.segment_name(rec)
        for system, items in by_system.items():
            if (name, system) in seen:
                skipped["outputs of a segment and system already counted (first one kept)"] += 1
                continue
            known = [it for it in items if it[1] is not None]
            if not known:
                skipped["outputs with null gold"] += 1
                continue
            g, d = known[0][1], known[0][2]
            text = next((t.get("text") for t, _, _, _ in items if t.get("text") is not None), None)
            if text is None:
                skipped["outputs without text"] += 1
                continue
            human = []
            if g:
                raw = next((hs for _, _, _, hs in known if isinstance(hs, list) and hs), None)
                if raw is None:
                    skipped["harmful outputs without gold_spans (not counted at all)"] += 1
                    continue
                for sp in raw:
                    pos = place(text, sp.get("start"), sp.get("end"), sp.get("text"))
                    if pos:
                        human.append(pos)
                    else:
                        skipped["human spans not found in the text (ignored)"] += 1
                if not human:
                    skipped["harmful outputs without a locatable human span (not counted at all)"] += 1
                    continue
            llm = []
            for t, _, _, _ in items:
                if t.get("span") is None:
                    continue
                pos = place(text, t.get("span_start"), t.get("span_end"), t.get("span"))
                if pos:
                    llm.append(pos)
                else:
                    skipped["LLM spans not found in the text (ignored)"] += 1
            seen.add((name, system))
            units.append({
                "segment": name, "document": rec.get("document"), "duration": evaluation.segment_duration(rec),
                "system": system, "asr": system == rec.get("asr_system"),
                "kind": "harmful" if g else "undecidable" if d == "undecidable" else "no_harm",
                "n_chars": len(text), "words": [m.span() for m in WORD.finditer(text)],
                "human": human, "llm": llm})
    return units, skipped, saw_spans


def make_groups(units):
    groups, is_asr = OrderedDict(), {}
    for u in units:
        is_asr[u["system"]] = is_asr.get(u["system"], False) or u["asr"]
        for group in (u["system"], None if u["asr"] else "all translations", "overall"):
            if group:
                groups.setdefault(group, []).append(u)
    systems = [g for g in groups if g not in ("all translations", "overall")]
    order = ([s for s in systems if is_asr.get(s)] + [s for s in systems if not is_asr.get(s)]
             + (["all translations"] if sum(not is_asr.get(s) for s in systems) > 1 else [])
             + ["overall"])
    return groups, is_asr, order


def summarize(units):
    """Counter of the totals, and the lists of characters and words of every human span."""
    S, span_chars, span_words = Counter(), [], []
    for u in units:
        words = u["words"]
        h, l = merge(u["human"]), merge(u["llm"])
        hw = {k for k, w in enumerate(words) if touches(w, h)}
        lw = {k for k, w in enumerate(words) if touches(w, l)}
        S["outputs"] += 1
        S[u["kind"]] += 1
        S["chars"] += u["n_chars"]
        S["words"] += len(words)
        S["h_spans"] += len(u["human"])
        S["h_chars"] += length(h)
        S["h_words"] += len(hw)
        S["l_spans"] += len(u["llm"])
        S["l_chars"] += length(l)
        S["l_words"] += len(lw)
        S["both_chars"] += intersection(h, l)
        S["both_words"] += len(hw & lw)
        S["h_spans_hit"] += sum(1 for sp in u["human"] if touches(sp, l))
        S["l_spans_hit"] += sum(1 for sp in u["llm"] if touches(sp, h))
        S["flagged"] += bool(u["llm"])
        S["harmful_flagged"] += bool(u["llm"]) and u["kind"] == "harmful"
        for sp in u["human"]:
            span_chars.append(sp[1] - sp[0])
            span_words.append(sum(1 for w in words if touches(w, [sp])))
    return S, span_chars, span_words


# ---------------------------------------------------------------- output

def span_stats(values):
    if not values:
        return ""
    return f"per span: mean {statistics.mean(values):.1f}, median {statistics.median(values):g}, max {max(values)}"


def print_group(label, units, out):
    S, span_chars, span_words = summarize(units)
    print(f"\n{label}  ({S['outputs']} outputs, {S['chars']:,} characters, {S['words']:,} words)", file=out)
    print(f"  outputs with a harmful span {S['harmful']} ({pct(S['harmful'], S['outputs'])}) | "
          f"no_harm {S['no_harm']} | undecidable {S['undecidable']}", file=out)
    per_output = f", {S['h_spans'] / S['harmful']:.2f} per harmful output" if S["harmful"] else ""
    print(f"  human harmful spans: {S['h_spans']}{per_output}", file=out)
    for what, key, total, values in (("characters", "h_chars", "chars", span_chars),
                                     ("words", "h_words", "words", span_words)):
        print(f"    {what:<11}{S[key]:>8,} = {pct(S[key], S[total]):>6} of all {what:<11}  "
              f"{span_stats(values)}".rstrip(), file=out)
    print(f"  LLM-flagged spans: {S['l_spans']} in {S['flagged']} outputs", file=out)
    for what, key, total in (("characters", "l_chars", "chars"), ("words", "l_words", "words")):
        print(f"    {what:<11}{S[key]:>8,} = {pct(S[key], S[total]):>6} of all {what}", file=out)
    print(f"  coverage       {'human':>8}{'covered':>9}{'recall':>8}{'95% CI':>14}  |"
          f"{'LLM':>8}{'correct':>9}{'precision':>11}", file=out)
    rows = (("characters", S["h_chars"], S["both_chars"], S["l_chars"], S["both_chars"], False),
            ("words", S["h_words"], S["both_words"], S["l_words"], S["both_words"], False),
            ("spans", S["h_spans"], S["h_spans_hit"], S["l_spans"], S["l_spans_hit"], True),
            ("outputs", S["harmful"], S["harmful_flagged"], S["flagged"], S["harmful_flagged"], True))
    for what, h_n, h_hit, l_n, l_hit, with_ci in rows:
        print(f"    {what:<11}{h_n:>8,}{h_hit:>9,}{rate(h_hit, h_n):>8}"
              f"{(ci_text(h_hit, h_n) if with_ci else ''):>14}  |"
              f"{l_n:>8,}{l_hit:>9,}{rate(l_hit, l_n):>11}", file=out)


def print_summary(groups, is_asr, order, out):
    print(f"\n{'summary':<24}{'outputs':>8}{'harmful':>8}{'spans':>7}{'chars':>8}{'%chars':>8}"
          f"{'words':>7}{'%words':>8} |{'rec.char':>9}{'rec.word':>9}{'rec.span':>9}{'rec.out':>8}", file=out)
    for group in order:
        if group not in groups:
            continue
        S, _, _ = summarize(groups[group])
        label = f"{group} (ASR)" if is_asr.get(group) else group
        print(f"{label:<24}{S['outputs']:>8}{S['harmful']:>8}{S['h_spans']:>7}{S['h_chars']:>8}"
              f"{pct(S['h_chars'], S['chars']):>8}{S['h_words']:>7}{pct(S['h_words'], S['words']):>8} |"
              f"{rate(S['both_chars'], S['h_chars']):>9}{rate(S['both_words'], S['h_words']):>9}"
              f"{rate(S['h_spans_hit'], S['h_spans']):>9}{rate(S['harmful_flagged'], S['harmful']):>8}", file=out)


def print_audio(units, out):
    segments = OrderedDict()
    for u in units:
        seg = segments.setdefault(u["segment"], {"duration": u["duration"], "harm": False})
        seg["harm"] = seg["harm"] or u["kind"] == "harmful"
    known = [s["duration"] for s in segments.values() if s["duration"] is not None]
    n_harm = sum(1 for s in segments.values() if s["harm"])
    print(f"\nannotated: {len(segments)} segments"
          + (f", audio {evaluation.fmt_duration(sum(known))}" if known else "")
          + f"; {n_harm} ({pct(n_harm, len(segments))}) with a harmful span {ci_text(n_harm, len(segments))}",
          file=out)


def original_summary(units, groups, is_asr, order, segments, duplicates, out):
    names = {u["segment"] for u in units}
    found = names & set(segments)
    all_dur = sum(x["duration"] or 0 for x in segments.values())
    ann_dur = sum(segments[n]["duration"] or 0 for n in found)
    n_docs = len({x["document"] for x in segments.values()})
    print("\noriginal data (--input-segments)", file=out)
    print(f"  input      {len(segments):>7} segments, {n_docs:>4} documents, audio "
          f"{evaluation.fmt_duration(all_dur)}", file=out)
    print(f"  annotated  {len(found):>7} segments ({evaluation.per(len(found), len(segments), 100)}% of the "
          f"input), audio {evaluation.fmt_duration(ann_dur)} ({evaluation.per(ann_dur, all_dur, 100)}%)",
          file=out)
    if names - found:
        print(f"  WARNING: {len(names - found)} evaluated segments are not in --input-segments", file=out)
    if duplicates:
        print(f"  note: {duplicates} duplicate segments in --input-segments counted once", file=out)
    if not found:
        return
    scale, h_ann = len(segments) / len(found), evaluation.hours(ann_dur)
    print(f"\nharmful spans and words per hour of annotated audio and per 100 annotated segments; 'est. input' "
          f"= count x {scale:.1f}\n(the estimate is valid only if the annotated segments are a random sample "
          f"of the input)", file=out)
    print(f"{'':<24}{'spans':>7}{'words':>7}{'spans/h':>9}{'words/h':>9}{'spans/100':>10}{'words/100':>10}"
          f"{'est.spans':>11}{'est.words':>11}", file=out)
    for group in order:
        if group not in groups:
            continue
        S, _, _ = summarize(groups[group])
        label = f"{group} (ASR)" if is_asr.get(group) else group
        print(f"{label:<24}{S['h_spans']:>7}{S['h_words']:>7}{evaluation.per(S['h_spans'], h_ann):>9}"
              f"{evaluation.per(S['h_words'], h_ann):>9}{evaluation.per(S['h_spans'], len(found), 100):>10}"
              f"{evaluation.per(S['h_words'], len(found), 100):>10}"
              f"{S['h_spans'] * scale:>11.0f}{S['h_words'] * scale:>11.0f}", file=out)
    n_harm = len({u["segment"] for u in units if u["kind"] == "harmful" and u["segment"] in found})
    print(f"segments with a harmful span: {n_harm} of {len(found)} annotated ({pct(n_harm, len(found))}, "
          f"95% CI {ci_text(n_harm, len(found))}); est. {n_harm * scale:.0f} of {len(segments)} input segments",
          file=out)


def report(records, title=None, out=sys.stdout, segments=None, duplicates=0):
    units, skipped, saw_spans = build_units(records)
    if records and not saw_spans:
        sys.exit("error: no record has gold_spans (the human error spans); generate the input again with the "
                 "current pearmut_to_jsonl.py")
    groups, is_asr, order = make_groups(units)
    print("=" * 72, file=out)
    print(f"HARMFUL SPANS{': ' + title if title else ''}  ({len(records)} segments)", file=out)
    print("=" * 72, file=out)
    if not units:
        print("  no outputs with a usable gold annotation", file=out)
    for group in order:
        if group in groups:
            print_group(f"{group} (ASR)" if is_asr.get(group) else group, groups[group], out)
    if units:
        print_summary(groups, is_asr, order, out)
        print_audio(units, out)
        if segments is not None:
            original_summary(units, groups, is_asr, order, segments, duplicates, out)
    print(file=out)
    for k, v in sorted(skipped.items()):
        print(f"  skipped: {k}: {v}", file=out)
    print("  note: the numbers describe the annotated segments; they estimate the whole data only if these "
          "segments are a random sample", file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of pearmut_to_jsonl.py (merged)")
    ap.add_argument("--input-segments", nargs="+",
                    help="aligned JSONL of all segments of the original documents (merged)")
    args = ap.parse_args()
    segments, duplicates = (evaluation.load_segments(args.input_segments) if args.input_segments
                            else (None, 0))
    report(evaluation.load(args.files), title=", ".join(args.files), segments=segments,
           duplicates=duplicates)


if __name__ == "__main__":
    main()
