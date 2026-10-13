#!/usr/bin/env python3
"""Inter-annotator agreement on the harmful-error annotation.

    python3 eval-iaa.py gold/Kimi_cs.jsonl [more files ...] [--annotators A B ...] [--list]

Input: the output of pearmut_to_jsonl.py made WITHOUT {user} in -o, from pearmut files with several
annotators (the field "gold_by_annotator" holds the verdict and the marked spans of every annotator).

The unit is the output of one system (the ASR or a translation) in one segment. The verdict of an annotator
on it is one of
  harmful      the annotator marked at least one error span in it
  no_harm      the annotator selected [no harm]
  undecidable  the annotator selected [undecidable]
An annotator who saw the output but marked nothing at all has no verdict (as in pearmut_to_jsonl.py). Only
units with verdicts of at least two annotators can be compared. The same unit in several input files is
merged (a rating of the same annotator is taken from the first file).

For each system, for all translations together and overall, it prints
  - per annotator: how many outputs were rated, their verdicts, and the spans they marked (number, mean
    length in characters and words),
  - for each pair of annotators on the outputs both rated: the confusion matrix of the verdicts, the observed
    agreement (95% Wilson interval), Cohen's kappa (approximate 95% interval) and Gwet's AC1, for the three
    classes and for harmful vs. other (the binary view is the gold of eval.py: undecidable counts as no
    harm),
  - Krippendorff's alpha (nominal) over all annotators, which also uses units rated by only some of them,
  - the agreement on the marked spans, on the outputs both rated: how many characters and words each marked,
    both marked, Dice and IoU of these sets, character- and word-level observed agreement and kappa (over
    all the text, so the unmarked majority counts as agreement), and the overlap-based F1 of the spans; and
    the same Dice and F1 restricted to the outputs both found harmful.
Words are whitespace-separated tokens, and a word is marked if it shares a character with a span.

Kappa is 0 (or undefined) when one annotator uses a single category, even if they agree often; AC1 and the
observed agreement are less sensitive to that. With few shared outputs all the intervals are wide.

--annotators: use only these annotators.
--list: also print the outputs where the verdicts differ, and where both found harm but marked different
        spans (Dice below 0.5), with the text and what each annotator marked.
"""
import argparse
import itertools
import math
import re
import statistics
import sys
from collections import Counter, OrderedDict

import eval as evaluation  # eval.py next to this script

KINDS = ("harmful", "no_harm", "undecidable")
WORD = re.compile(r"\S+")


# ---------------------------------------------------------------- intervals (as in eval-recall.py)

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


# ---------------------------------------------------------------- agreement statistics

def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - width) / denom, (centre + width) / denom


def f3(v):
    return "    -" if v is None else f"{v:.3f}"


def ci(c):
    return "" if c is None else f"[{c[0]:.2f},{c[1]:.2f}]"


def agreement(pairs, cats):
    """Observed agreement, Cohen's kappa (+ approximate interval) and Gwet's AC1 of a list of (a, b) labels."""
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    kappa = (po - pe) / (1 - pe) if pe < 1 else None
    kappa_ci = None
    if kappa is not None:
        se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
        kappa_ci = (kappa - 1.96 * se, kappa + 1.96 * se)
    pi = {c: (pa[c] + pb[c]) / 2 for c in cats}
    pe_g = sum(pi[c] * (1 - pi[c]) for c in cats) / (len(cats) - 1)
    ac1 = (po - pe_g) / (1 - pe_g) if pe_g < 1 else None
    return {"n": n, "po": po, "po_ci": wilson(sum(1 for a, b in pairs if a == b), n),
            "kappa": kappa, "kappa_ci": kappa_ci, "ac1": ac1}


def kappa_from_counts(n11, n10, n01, n00):
    """Cohen's kappa of two binary labelings given the 2x2 counts; None if undefined."""
    n = n11 + n10 + n01 + n00
    if not n:
        return None
    po = (n11 + n00) / n
    pe = ((n11 + n10) / n) * ((n11 + n01) / n) + ((n01 + n00) / n) * ((n10 + n00) / n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def krippendorff_alpha(rating_lists):
    """Nominal Krippendorff's alpha of a list of units, each a list of the labels given to it."""
    o, n_c = Counter(), Counter()
    for ratings in rating_lists:
        m = len(ratings)
        if m < 2:
            continue
        for i, j in itertools.permutations(range(m), 2):
            o[(ratings[i], ratings[j])] += 1 / (m - 1)
    for (c, _), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return None
    observed = sum(v for (c, k), v in o.items() if c != k)
    expected = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    return None if expected == 0 else 1 - (n - 1) * observed / expected


def dice(both, a, b):
    return 2 * both / (a + b) if a + b else None


def iou(both, a, b):
    return both / (a + b - both) if a + b - both else None


# ---------------------------------------------------------------- units

def build_units(records, only):
    """Units (segment, system) with the rating of every annotator: {"kind", "spans" (located) or None}."""
    units, skipped, saw = OrderedDict(), Counter(), False
    for rec in records:
        targets, by_annotator = rec.get("targets", []), rec.get("gold_by_annotator")
        if not isinstance(by_annotator, dict):
            skipped["records without gold_by_annotator"] += 1
            continue
        saw = True
        name = evaluation.segment_name(rec)
        by_system = OrderedDict()
        for k, t in enumerate(targets):
            by_system.setdefault(t.get("system"), []).append(k)
        for system, idx in by_system.items():
            text = next((targets[k].get("text") for k in idx if targets[k].get("text") is not None), None)
            if text is None:
                skipped["outputs without text"] += 1
                continue
            for user, ann in by_annotator.items():
                if only and user not in only:
                    continue
                gold, details, spans = (ann.get("gold_annotations"), ann.get("gold_annotation_details"),
                                        ann.get("gold_spans"))
                if not isinstance(gold, list) or len(gold) != len(targets):
                    skipped["annotator entries not matching the targets"] += 1
                    continue
                if not isinstance(details, list) or len(details) != len(targets):
                    details = [None] * len(targets)
                if not isinstance(spans, list) or len(spans) != len(targets):
                    spans = [None] * len(targets)
                k = next((k for k in idx if gold[k] is not None), None)
                if k is None:
                    continue  # this annotator did not see the output (or marked nothing)
                kind = "harmful" if gold[k] else "undecidable" if details[k] == "undecidable" else "no_harm"
                located = []
                if kind == "harmful":
                    raw = next((spans[j] for j in idx if gold[j] is not None and isinstance(spans[j], list)
                                and spans[j]), None)
                    if raw is None:
                        skipped["harmful ratings without gold_spans (left out of the span analysis)"] += 1
                        located = None
                    else:
                        for sp in raw:
                            pos = place(text, sp.get("start"), sp.get("end"), sp.get("text"))
                            if pos:
                                located.append(pos)
                            else:
                                skipped["marked spans not found in the text (ignored)"] += 1
                        if not located:
                            located = None
                unit = units.setdefault((name, system), {
                    "segment": name, "document": rec.get("document"), "system": system,
                    "asr": system == rec.get("asr_system"), "text": text, "n_chars": len(text),
                    "words": [m.span() for m in WORD.finditer(text)], "ratings": OrderedDict()})
                if user in unit["ratings"]:
                    skipped["ratings of the same annotator, segment and system again (first kept)"] += 1
                    continue
                unit["ratings"][user] = {"kind": kind, "spans": located}
    return list(units.values()), skipped, saw


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


# ---------------------------------------------------------------- span agreement

def span_counts(units, a, b):
    """Characters, words and spans marked by annotators a and b in the units both rated and located."""
    S = Counter()
    for u in units:
        sa, sb = u["ratings"][a]["spans"], u["ratings"][b]["spans"]
        if sa is None or sb is None:
            S["left_out"] += 1
            continue
        ha, hb = merge(sa), merge(sb)
        wa = {k for k, w in enumerate(u["words"]) if touches(w, ha)}
        wb = {k for k, w in enumerate(u["words"]) if touches(w, hb)}
        S["units"] += 1
        S["chars"] += u["n_chars"]
        S["words"] += len(u["words"])
        S["a_chars"], S["b_chars"] = S["a_chars"] + length(ha), S["b_chars"] + length(hb)
        S["ab_chars"] += intersection(ha, hb)
        S["a_words"], S["b_words"] = S["a_words"] + len(wa), S["b_words"] + len(wb)
        S["ab_words"] += len(wa & wb)
        S["a_spans"], S["b_spans"] = S["a_spans"] + len(sa), S["b_spans"] + len(sb)
        S["a_hit"] += sum(1 for sp in sa if touches(sp, hb))
        S["b_hit"] += sum(1 for sp in sb if touches(sp, ha))
    return S


def span_f1(S):
    """Overlap-based F1: precision = spans of a that overlap a span of b, recall = the other way round."""
    p = S["a_hit"] / S["a_spans"] if S["a_spans"] else None
    r = S["b_hit"] / S["b_spans"] if S["b_spans"] else None
    return 2 * p * r / (p + r) if p is not None and r is not None and p + r else None


def print_span_agreement(shared, a, b, out):
    S = span_counts(shared, a, b)
    if not S["units"]:
        print("    spans: no shared output with located spans", file=out)
        return
    print(f"    marked text on the {S['units']} shared outputs (a = {a}, b = {b})", file=out)
    print(f"      {'':<12}{'a':>8}{'b':>8}{'both':>8}{'Dice':>8}{'IoU':>8}{'agree':>8}{'kappa':>8}", file=out)
    for what, ka, kb, kab, total in (("characters", "a_chars", "b_chars", "ab_chars", "chars"),
                                     ("words", "a_words", "b_words", "ab_words", "words")):
        n11, n10, n01 = S[kab], S[ka] - S[kab], S[kb] - S[kab]
        n00 = S[total] - n11 - n10 - n01
        print(f"      {what:<12}{S[ka]:>8}{S[kb]:>8}{S[kab]:>8}{f3(dice(S[kab], S[ka], S[kb])):>8}"
              f"{f3(iou(S[kab], S[ka], S[kb])):>8}{f3((n11 + n00) / S[total]):>8}"
              f"{f3(kappa_from_counts(n11, n10, n01, n00)):>8}", file=out)
    print(f"      spans: a marked {S['a_spans']}, b marked {S['b_spans']}; {S['a_hit']} of a's overlap a span of b, "
          f"{S['b_hit']} of b's overlap a span of a; overlap F1 {f3(span_f1(S)).strip()}", file=out)
    both = [u for u in shared if u["ratings"][a]["kind"] == "harmful" and u["ratings"][b]["kind"] == "harmful"]
    B = span_counts(both, a, b)
    if B["units"]:
        print(f"      only the {B['units']} outputs both found harmful: Dice characters "
              f"{f3(dice(B['ab_chars'], B['a_chars'], B['b_chars'])).strip()}, words "
              f"{f3(dice(B['ab_words'], B['a_words'], B['b_words'])).strip()}, overlap F1 "
              f"{f3(span_f1(B)).strip()}", file=out)
    if S["left_out"]:
        print(f"      ({S['left_out']} shared outputs without located spans left out)", file=out)


# ---------------------------------------------------------------- output

def verdict_columns(pairs):
    """Agreement for three classes and for harmful vs. other."""
    three = agreement(pairs, KINDS)
    two = agreement([("harmful" if a == "harmful" else "other", "harmful" if b == "harmful" else "other")
                     for a, b in pairs], ("harmful", "other"))
    return three, two


def print_pair(a, b, shared, out):
    pairs = [(u["ratings"][a]["kind"], u["ratings"][b]["kind"]) for u in shared]
    w = max(len(a), len(b), 11) + 2
    print(f"\n  {a} (rows) vs. {b} (columns): {len(shared)} shared outputs", file=out)
    print(f"    {'':<{w}}" + "".join(f"{k:>13}" for k in KINDS), file=out)
    for ka in KINDS:
        print(f"    {ka:<{w}}" + "".join(f"{sum(1 for p in pairs if p == (ka, kb)):>13}" for kb in KINDS),
              file=out)
    three, two = verdict_columns(pairs)
    print(f"    {'':<22}{'3 classes':>22}{'harmful vs. other':>26}", file=out)
    print(f"    {'observed agreement':<22}{f3(three['po']):>8} {ci(three['po_ci']):>13}"
          f"{f3(two['po']):>13} {ci(two['po_ci']):>13}", file=out)
    print(f"    {'Cohen kappa':<22}{f3(three['kappa']):>8} {ci(three['kappa_ci']):>13}"
          f"{f3(two['kappa']):>13} {ci(two['kappa_ci']):>13}", file=out)
    print(f"    {'Gwet AC1':<22}{f3(three['ac1']):>8}{'':>14}{f3(two['ac1']):>13}", file=out)
    if three["kappa"] is None or two["kappa"] is None or two["kappa"] == 0:
        print("    note: kappa is undefined or 0 when an annotator uses a single category; see the observed "
              "agreement and AC1", file=out)
    return three, two


def print_group(label, units, annotators, out, summary):
    rated = Counter()
    for u in units:
        rated.update(u["ratings"].keys())
    shared = [u for u in units if len(u["ratings"]) >= 2]
    print(f"\n{label}  ({len(units)} outputs rated, {len(shared)} by two or more annotators)", file=out)
    width = max([len(a) for a in annotators] + [10]) + 2
    print(f"  {'annotator':<{width}}{'rated':>7}{'harmful':>9}{'no_harm':>9}{'undecidable':>13}"
          f"{'spans':>7}{'chars/span':>12}{'words/span':>12}", file=out)
    for a in annotators:
        mine = [u["ratings"][a] for u in units if a in u["ratings"]]
        if not mine:
            continue
        kinds = Counter(r["kind"] for r in mine)
        chars, words = [], []
        for u in units:
            r = u["ratings"].get(a)
            if r and r["spans"]:
                for sp in r["spans"]:
                    chars.append(sp[1] - sp[0])
                    words.append(sum(1 for w in u["words"] if touches(w, [sp])))
        print(f"  {a:<{width}}{len(mine):>7}{kinds['harmful']:>9}{kinds['no_harm']:>9}{kinds['undecidable']:>13}"
              f"{len(chars):>7}{(f'{statistics.mean(chars):.1f}' if chars else '-'):>12}"
              f"{(f'{statistics.mean(words):.1f}' if words else '-'):>12}", file=out)
    if not shared:
        print("  no output was rated by two annotators: nothing to compare", file=out)
        return
    for a, b in itertools.combinations(annotators, 2):
        both = [u for u in shared if a in u["ratings"] and b in u["ratings"]]
        if not both:
            continue
        three, two = print_pair(a, b, both, out)
        print_span_agreement(both, a, b, out)
        S = span_counts(both, a, b)
        summary.append((label, a, b, three, two, dice(S["ab_chars"], S["a_chars"], S["b_chars"])))
    lists = [[r["kind"] for r in u["ratings"].values()] for u in shared]
    binary = [["harmful" if k == "harmful" else "other" for k in ratings] for ratings in lists]
    print(f"\n  Krippendorff's alpha over all annotators ({len(shared)} outputs): 3 classes "
          f"{f3(krippendorff_alpha(lists)).strip()}, harmful vs. other {f3(krippendorff_alpha(binary)).strip()}",
          file=out)


def mark(text, spans):
    pieces, pos = [], 0
    for s, e in merge(spans or []):
        pieces.append(text[pos:s] + "«" + text[s:e] + "»")
        pos = e
    return "".join(pieces) + text[pos:]


def print_list(units, annotators, out):
    differ, spans_differ = [], []
    for u in units:
        if len(u["ratings"]) < 2:
            continue
        kinds = {r["kind"] for r in u["ratings"].values()}
        if len(kinds) > 1:
            differ.append(u)
            continue
        if kinds == {"harmful"} and all(r["spans"] for r in u["ratings"].values()):
            for a, b in itertools.combinations(u["ratings"], 2):
                S = span_counts([u], a, b)
                d = dice(S["ab_chars"], S["a_chars"], S["b_chars"])
                if d is not None and d < 0.5:
                    spans_differ.append((u, a, b, d))
                    break
    print(f"\nOUTPUTS WITH DIFFERENT VERDICTS: {len(differ)}", file=out)
    for u in differ:
        print(f"\n  {u['segment']}  {u['system']}", file=out)
        for a, r in u["ratings"].items():
            print(f"    {a}: {r['kind']}   {mark(u['text'], r['spans']) if r['spans'] else ''}", file=out)
    print(f"\nBOTH HARMFUL BUT DIFFERENT SPANS (Dice < 0.5): {len(spans_differ)}", file=out)
    for u, a, b, d in spans_differ:
        print(f"\n  {u['segment']}  {u['system']}  (Dice {d:.2f})", file=out)
        for who in (a, b):
            print(f"    {who}: {mark(u['text'], u['ratings'][who]['spans'])}", file=out)


def print_summary(summary, out):
    if not summary:
        return
    w1 = max(len(r[0]) for r in summary) + 2
    w2 = max(len(r[1]) + len(r[2]) + 3 for r in summary) + 2
    print(f"\n{'summary':<{w1}}{'annotator pair':<{w2}}{'n':>4} |{'agree':>7}{'kappa':>7}{'AC1':>7} |"
          f"{'agree':>7}{'kappa':>7}{'AC1':>7} |{'Dice':>7}", file=out)
    print(f"{'':<{w1}}{'':<{w2}}{'':>4} |{'3 classes':>21} |{'harmful vs. other':>21} |{'chars':>7}", file=out)
    for label, a, b, three, two, d in summary:
        print(f"{label:<{w1}}{a + ' / ' + b:<{w2}}{three['n']:>4} |{f3(three['po']):>7}"
              f"{f3(three['kappa']):>7}{f3(three['ac1']):>7} |{f3(two['po']):>7}{f3(two['kappa']):>7}"
              f"{f3(two['ac1']):>7} |{f3(d):>7}", file=out)


def report(records, title=None, out=sys.stdout, only=None, list_all=False):
    units, skipped, saw = build_units(records, only)
    if records and not saw:
        sys.exit("error: no record has gold_by_annotator; generate the input again with the current "
                 "pearmut_to_jsonl.py (without {user} in -o)")
    annotators = sorted({a for u in units for a in u["ratings"]})
    groups, is_asr, order = make_groups(units)
    print("=" * 72, file=out)
    print(f"INTER-ANNOTATOR AGREEMENT{': ' + title if title else ''}  ({len(records)} segments)", file=out)
    print("=" * 72, file=out)
    print(f"annotators: {', '.join(annotators) or '-'}", file=out)
    summary = []
    for group in order:
        if group in groups:
            print_group(f"{group} (ASR)" if is_asr.get(group) else group, groups[group], annotators, out,
                        summary)
    print_summary(summary, out)
    if list_all:
        print_list(units, annotators, out)
    print(file=out)
    for k, v in sorted(skipped.items()):
        print(f"  skipped: {k}: {v}", file=out)
    print("  note: only outputs with a verdict of at least two annotators are compared; an annotator who marked "
          "nothing for an output has no verdict on it", file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of pearmut_to_jsonl.py (merged)")
    ap.add_argument("--annotators", nargs="+", help="use only these annotators")
    ap.add_argument("--list", action="store_true",
                    help="also list the outputs with different verdicts or different spans")
    args = ap.parse_args()
    report(evaluation.load(args.files), title=", ".join(args.files),
           only=set(args.annotators) if args.annotators else None, list_all=args.list)


if __name__ == "__main__":
    main()
