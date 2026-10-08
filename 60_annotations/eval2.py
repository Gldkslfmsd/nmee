#!/usr/bin/env python3
"""Does the LLM's own harm rating help, and what kinds of errors does it flag? (translations only)

    python3 eval2.py gold/Kimi_cs.jsonl [more files ...] [--min-kept 10]

Input: output of pearmut_to_jsonl.py (records with "targets" and "gold_annotations"). Same decision
model as eval.py, but the ASR system ("asr_system") is left out and only translation targets count:
  flagged      target with a span (the LLM flagged it); gold true = TP, gold false = FP
  not flagged  target without a span; gold true = FN, gold false = TN
  gold null    skipped
  undecidable  targets whose "gold_annotation_details" is "undecidable" are skipped too; the number of
               flagged and unflagged ones is printed at the end

Three parts:
  1. overview      confusion matrix and precision/recall/F1 of the translations, overall and per system
  2. harmfulness   would keeping only flags with harmfulness >= t give better precision? Precision, recall
                   and F1 for every threshold t, precision per exact harmfulness level, and AUC of the
                   harmfulness score for separating human-confirmed errors (TP) from false alarms (FP)
  3. categories    precision and share of confirmed errors / false alarms by error_source, harm_types
                   (a flag with several types counts in each), combinations of types and target language

Precision has a 95% Wilson interval: with few annotated flags, differences between thresholds or
categories are often within noise; the script says so when intervals overlap.

Caveats: only LLM-preselected segments were annotated, so recall is recall within those segments. FN have
no harmfulness or categories (the LLM did not flag them), so the categories describe flagged spans only.
"""
import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, OrderedDict


# ---------------------------------------------------------------- loading

def load(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    return records


def number(value):
    """A numeric rating or None (bool and strings that are not numbers are not ratings)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def harm_types(target):
    types = target.get("harm_types")
    if isinstance(types, str):
        types = [types]
    types = [str(t).strip() for t in (types or []) if str(t).strip()]
    return types or ["(none)"]


def collect(records):
    """Rows of translation targets with a gold label, and a Counter of what was skipped."""
    rows, skipped = [], Counter()
    for rec in records:
        targets, gold = rec.get("targets", []), rec.get("gold_annotations")
        if gold is None or len(gold) != len(targets):
            skipped["records without matching gold_annotations"] += 1
            continue
        details = rec.get("gold_annotation_details")
        if details is None or len(details) != len(targets):
            details = [None] * len(targets)  # older files without details: nothing is undecidable
        for t, g, d in zip(targets, gold, details):
            if t.get("system") == rec.get("asr_system"):
                skipped["ASR targets (not evaluated here)"] += 1
                continue
            if g is None:
                skipped["targets with null gold"] += 1
                continue
            flagged = t.get("span") is not None
            if d == "undecidable":
                skipped["flagged translation targets with undecidable gold" if flagged
                        else "unflagged translation targets with undecidable gold"] += 1
                continue
            rows.append({
                "system": t.get("system"),
                "lan": t.get("tgt_lan") or "?",
                "flagged": flagged,
                "gold": bool(g),
                "harm": number(t.get("harmfulness")) if flagged else None,
                "types": harm_types(t) if flagged else [],
                "source": str(t.get("error_source") or "(none)") if flagged else None,
            })
    return rows, skipped


# ---------------------------------------------------------------- statistics

def ratio(a, b):
    return a / b if b else None


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - width) / denom, (centre + width) / denom


def fmt(v):
    return "    -" if v is None else f"{v:.3f}"


def fmt_ci(ci):
    return "      -      " if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]"


def overlap(a, b):
    return a is not None and b is not None and a[0] <= b[1] and b[0] <= a[1]


def f1_of(p, r):
    return 2 * p * r / (p + r) if p is not None and r is not None and p + r else None


def confusion(rows):
    c = Counter()
    for r in rows:
        c["tp" if r["flagged"] and r["gold"] else "fp" if r["flagged"] else "fn" if r["gold"] else "tn"] += 1
    return c


def auc(positive, negative):
    """P(random positive scores higher than random negative), ties count half."""
    if not positive or not negative:
        return None
    total = sum(1 if a > b else 0.5 if a == b else 0 for a in positive for b in negative)
    return total / (len(positive) * len(negative))


def mean(values):
    return statistics.mean(values) if values else None


def label(x):
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


# ---------------------------------------------------------------- 1. overview

def overview(rows, out):
    print("\n1. OVERVIEW (translations only)", file=out)
    print(f"{'':<24}{'n':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}"
          f"{'acc':>8}{'prec':>8}{'rec':>8}{'F1':>8}", file=out)
    systems = list(OrderedDict.fromkeys(r["system"] for r in rows))
    groups = [(s, [r for r in rows if r["system"] == s]) for s in systems]
    if len(systems) > 1:
        groups.append(("all translations", rows))
    for name, group in groups:
        c = confusion(group)
        n = sum(c.values())
        p, rec = ratio(c["tp"], c["tp"] + c["fp"]), ratio(c["tp"], c["tp"] + c["fn"])
        print(f"{name:<24}{n:>6}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}{c['tn']:>6}"
              + "".join(f"{fmt(v):>8}" for v in (ratio(c["tp"] + c["tn"], n), p, rec, f1_of(p, rec))),
              file=out)
    c = confusion(rows)
    ci = wilson(c["tp"], c["tp"] + c["fp"])
    print(f"  precision 95% CI {fmt_ci(ci)}, flags {c['tp'] + c['fp']}, "
          f"human-confirmed errors {c['tp'] + c['fn']} (of which flagged by the LLM {c['tp']})", file=out)


# ---------------------------------------------------------------- 2. harmfulness

def harmfulness(rows, min_kept, out):
    print("\n2. DOES FILTERING BY HARMFULNESS IMPROVE PRECISION?", file=out)
    flagged = [r for r in rows if r["flagged"]]
    if not flagged:
        print("  no flagged targets", file=out)
        return
    c = confusion(rows)
    missed = c["fn"]  # confirmed errors that were never flagged: lost at every threshold
    positives = c["tp"] + missed
    rated = [r for r in flagged if r["harm"] is not None]
    unrated = len(flagged) - len(rated)
    levels = sorted({r["harm"] for r in rated})
    if unrated:
        print(f"  note: {unrated} flags have no numeric harmfulness; they count only in 'all flags'",
              file=out)
    if not levels:
        print("  no flagged target has a numeric harmfulness", file=out)
        return

    print("\n  keep flags with harmfulness >= t   (recall over all human-confirmed errors in the annotated "
          "segments)", file=out)
    print(f"  {'t':<10}{'kept':>6}{'%kept':>7}{'TP':>6}{'FP':>6}{'prec':>8}{'95% CI':>14}"
          f"{'rec':>8}{'F1':>8}", file=out)
    table = [("all flags", flagged)] + [(f">= {label(t)}", [r for r in rated if r["harm"] >= t])
                                        for t in levels]
    results = []
    for name, kept in table:
        tp = sum(1 for r in kept if r["gold"])
        fp = len(kept) - tp
        p, rec = ratio(tp, len(kept)), ratio(tp, positives)
        ci = wilson(tp, len(kept))
        results.append({"name": name, "kept": len(kept), "p": p, "r": rec, "f1": f1_of(p, rec), "ci": ci})
        print(f"  {name:<10}{len(kept):>6}{ratio(len(kept), len(flagged)) * 100:>6.0f}%{tp:>6}{fp:>6}"
              f"{fmt(p):>8}  {fmt_ci(ci):>12}{fmt(rec):>8}{fmt(f1_of(p, rec)):>8}", file=out)

    print("\n  precision per exact harmfulness level", file=out)
    print(f"  {'level':<10}{'n':>6}{'TP':>6}{'FP':>6}{'prec':>8}{'95% CI':>14}", file=out)
    for level in levels:
        group = [r for r in rated if r["harm"] == level]
        tp = sum(1 for r in group if r["gold"])
        print(f"  {label(level):<10}{len(group):>6}{tp:>6}{len(group) - tp:>6}"
              f"{fmt(ratio(tp, len(group))):>8}  {fmt_ci(wilson(tp, len(group))):>12}", file=out)

    tp_h = [r["harm"] for r in rated if r["gold"]]
    fp_h = [r["harm"] for r in rated if not r["gold"]]
    a = auc(tp_h, fp_h)
    print("\n  summary", file=out)
    print(f"  mean harmfulness: confirmed errors (TP) {fmt(mean(tp_h)).strip()} (n={len(tp_h)}), "
          f"false alarms (FP) {fmt(mean(fp_h)).strip()} (n={len(fp_h)})", file=out)
    if a is None:
        print("  AUC: undefined (need both TP and FP)", file=out)
    else:
        meaning = ("no better than chance" if abs(a - 0.5) < 0.05
                   else "harmfulness ranks real errors higher" if a > 0.5
                   else "harmfulness ranks real errors LOWER than false alarms")
        print(f"  AUC of harmfulness (TP vs FP): {a:.3f}  (0.5 = chance; {meaning})", file=out)

    base = results[0]
    candidates = [x for x in results[1:] if x["kept"] >= min_kept and x["p"] is not None]
    if not candidates:
        print(f"  no threshold keeps at least {min_kept} flags (--min-kept): nothing to recommend", file=out)
        return
    best_p = max(candidates, key=lambda x: (x["p"], x["kept"]))
    best_f1 = max((x for x in [base] + candidates if x["f1"] is not None), key=lambda x: x["f1"],
                  default=None)
    gain = best_p["p"] - (base["p"] or 0)
    print(f"  highest precision with >= {min_kept} flags kept: harmfulness {best_p['name']} -> "
          f"precision {best_p['p']:.3f} vs {fmt(base['p']).strip()} for all flags "
          f"({gain:+.3f}), recall {fmt(best_p['r']).strip()} vs {fmt(base['r']).strip()}", file=out)
    if best_f1 is base:
        print(f"  best F1: keeping all flags ({base['f1']:.3f}); no threshold improves it", file=out)
    elif best_f1 is not None:
        print(f"  best F1 with >= {min_kept} flags kept: harmfulness {best_f1['name']} -> F1 "
              f"{best_f1['f1']:.3f} vs {fmt(base['f1']).strip()} for all flags", file=out)
    if gain <= 0:
        print("  verdict: filtering does not improve precision here", file=out)
    elif overlap(best_p["ci"], base["ci"]):
        print("  verdict: precision goes up, but the intervals overlap: not clearly better with this "
              "amount of data", file=out)
    else:
        print("  verdict: precision goes up and the intervals do not overlap; the price is the lost "
              "recall shown above", file=out)


# ---------------------------------------------------------------- 3. categories

def breakdown(flagged, keys_of, title, out, note=None, limit=None):
    """Table of flagged targets grouped by the keys returned by keys_of(row)."""
    groups = {}
    for r in flagged:
        for key in keys_of(r):
            groups.setdefault(key, []).append(r)
    total_tp = sum(1 for r in flagged if r["gold"])
    total_fp = len(flagged) - total_tp
    print(f"\n  {title}" + (f"   ({note})" if note else ""), file=out)
    print(f"  {'':<46}{'n':>5}{'TP':>5}{'FP':>5}{'prec':>7}{'95% CI':>14}{'%TP':>6}{'%FP':>6}{'harm':>6}",
          file=out)
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    for key, group in order[:limit]:
        tp = sum(1 for r in group if r["gold"])
        fp = len(group) - tp
        harm = mean([r["harm"] for r in group if r["harm"] is not None])
        print(f"  {str(key)[:45]:<46}{len(group):>5}{tp:>5}{fp:>5}{fmt(ratio(tp, len(group))):>7}"
              f"  {fmt_ci(wilson(tp, len(group))):>12}"
              f"{ratio(tp, total_tp) * 100 if total_tp else 0:>5.0f}%"
              f"{ratio(fp, total_fp) * 100 if total_fp else 0:>5.0f}%"
              f"{fmt(harm):>6}", file=out)
    if limit and len(order) > limit:
        print(f"  ... {len(order) - limit} more", file=out)


def categories(rows, out):
    print("\n3. CATEGORIES OF FLAGGED ERRORS", file=out)
    print("  n/TP/FP = flagged targets in the category; %TP/%FP = share of all confirmed errors / false "
          "alarms;", file=out)
    print("  harm = mean harmfulness. Errors the LLM missed (FN) have no categories and are not included.",
          file=out)
    flagged = [r for r in rows if r["flagged"]]
    if not flagged:
        print("  no flagged targets", file=out)
        return
    breakdown(flagged, lambda r: [r["source"]], "error_source", out)
    breakdown(flagged, lambda r: r["types"], "harm_types", out,
              note="a flag with several types counts in each, so rows add up to more than the flags")
    breakdown(flagged, lambda r: [" + ".join(sorted(r["types"]))], "combination of harm_types", out,
              limit=10)
    breakdown(flagged, lambda r: [f"{r['source']} / {t}" for t in r["types"]],
              "error_source / harm_type", out, limit=15)
    if len({r["lan"] for r in flagged}) > 1:
        breakdown(flagged, lambda r: [r["lan"]], "target language", out)
    if len({r["system"] for r in flagged}) > 1:
        breakdown(flagged, lambda r: [r["system"]], "system", out)


# ---------------------------------------------------------------- main

def report(records, title=None, min_kept=10, out=sys.stdout):
    rows, skipped = collect(records)
    print("=" * 76, file=out)
    print(f"EVALUATION OF TRANSLATIONS{': ' + title if title else ''}  ({len(records)} segments)", file=out)
    print("=" * 76, file=out)
    if not rows:
        print("  no translation targets with a gold annotation", file=out)
    else:
        overview(rows, out)
        harmfulness(rows, min_kept, out)
        categories(rows, out)
    print(file=out)
    for k, v in sorted(skipped.items()):
        print(f"  skipped: {k}: {v}", file=out)
    print("  note: only LLM-preselected segments were annotated; recall is within those segments", file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of pearmut_to_jsonl.py (merged)")
    ap.add_argument("--min-kept", type=int, default=10,
                    help="smallest number of flags a harmfulness threshold must keep to be recommended "
                         "(default 10)")
    args = ap.parse_args()
    report(load(args.files), title=", ".join(os.path.basename(p) for p in args.files),
           min_kept=args.min_kept)


if __name__ == "__main__":
    main()