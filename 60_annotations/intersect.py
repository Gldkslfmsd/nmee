#!/usr/bin/env python3
"""Intersect several LLM preselections: keep only the flagged spans that the LLM runs agree on.

    python3 intersect.py out/Kimi_cs.jsonl out/Qwen_cs.jsonl -o out/Kimi_Qwen_cs.jsonl
    python3 intersect.py out/A.jsonl out/B.jsonl out/C.jsonl --min-files 2 -o out/vote2.jsonl

Input: outputs of 40_find_harmful_errors.py (one line per segment, "targets" with flagged spans), e.g. the
same translations annotated by different LLMs or by different prompts. Segments are matched by the audio
file name, targets by "system". A segment missing from a file is a segment where that run flagged nothing.

A flagged target of one file is supported by another file if that file flagged, in the same segment and
system, a span that matches it (--match):
  overlap  the character ranges overlap (default; compared by offsets if the texts are identical, else the
           span is searched in the text); empty spans match when they touch
  exact    the same span (same text, same position)
  segment  any flagged span in the same segment and system
By default a target is kept only if all files support it; --min-files N keeps targets flagged by at least
N files (N = 1 is the union without duplicates).

Output: JSONL in the format of the inputs, so it can be the --input of pearmut_to_jsonl.py. A kept target
is the one of the first file (in the order of the arguments) that flagged it: its span, intended,
harm_types, harmfulness, explanation and error_source are that file's. It gets two more keys:
  "n_agree": number of files that flagged it
  "agreed_by": [{"file", "model", "span", "span_start", "span_end", "harmfulness", "harm_types",
                 "error_source"}, ...]   -- the matching target of every supporting file, to compare
                 e.g. the harmfulness ratings or to filter by their minimum or mean
and the record gets "intersected_from": [{"file", "model"}, ...]. Segments without a kept target are
dropped.
The ASR is always among the "targets" of an output record: if no flagged ASR span was kept, a target
without a span is appended ("tgt_lan" = src_language, "system" = asr_system, "text" = asr, "span" null),
the same as pearmut_to_jsonl.py would add; it is the ASR that the annotator is shown next to the translation.

Gold annotations: if the inputs are outputs of pearmut_to_jsonl.py (records with "gold_annotations" and
"gold_annotation_details", one per target), they are kept: the output record gets the same two lists,
rebuilt for the output targets, plus
  "gold_disagreement": [null, [{"file", "model", "gold_annotations", "gold_annotation_details"}, ...], ...]
one entry per target: null if the files agree, else the verdict of every file that has one. The gold of a
target is the verdict of the first file (in the order of the arguments, among the files that flagged it)
that has a non-null gold. Files disagree if their golds (true/false) or their details differ; a file
without a gold (null, or no gold at all) does not disagree. The gold of the ASR target is taken from the
first file that has one, and compared over all files that contain the segment. Gold is a verdict on the
whole output of a system, so with --match segment the files annotated by the same annotator normally agree.
Statistics go to stderr: flags per file, how many of them were kept, the gold disagreements and the
pairwise agreement.
"""
import argparse
import copy
import json
import os
import sys
from collections import defaultdict


def shout(msg):
    print(msg, file=sys.stderr)


def audio_name(path):
    return os.path.basename(path or "")


class Preselection:
    """One input file: its records and an index (audio name, system) -> flagged targets."""

    def __init__(self, path):
        self.path = path
        self.records, self.index, self.model = [], defaultdict(list), None
        self.gold_of = {}        # id(target) -> (gold, detail)
        self.system_gold = {}    # (audio name, system) -> (gold, detail) of its first target in the segment
        self.has_gold, self.bad_gold = False, 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.records.append(rec)
                self.model = self.model or (rec.get("annotator") or {}).get("model")
                name = audio_name(rec.get("audio"))
                targets = rec.get("targets", [])
                gold, details = rec.get("gold_annotations"), rec.get("gold_annotation_details")
                ok = isinstance(gold, list) and len(gold) == len(targets)
                if gold is not None and not ok:
                    self.bad_gold += 1  # gold of another length than targets: cannot be paired
                self.has_gold = self.has_gold or ok
                ok_details = ok and isinstance(details, list) and len(details) == len(targets)
                for k, t in enumerate(targets):
                    g = gold[k] if ok else None
                    d = details[k] if ok_details else None
                    self.gold_of[id(t)] = (g, d)
                    self.system_gold.setdefault((name, t.get("system")), (g, d))
                    if t.get("span") is not None:
                        self.index[(name, t.get("system"))].append(t)


def locate(target, text):
    """(start, end) of the target's span in `text`, or None if it cannot be placed there."""
    span = target.get("span")
    if span is None:
        return None
    if target.get("text") == text:
        s, e = target.get("span_start"), target.get("span_end")
        if s is not None and e is not None:
            return s, e
    i = text.find(span)
    return None if i < 0 else (i, i + len(span))


def overlaps(x, y):
    (s, e), (hs, he) = x, y
    if e == s:
        return hs <= s <= he
    if he == hs:
        return s <= hs <= e
    return hs < e and s < he


def match(a, b, mode):
    """Do two flagged targets of the same segment and system agree?"""
    if mode == "segment":
        return True
    text = a.get("text") or ""
    la, lb = locate(a, text), locate(b, text)
    if la is None or lb is None:
        return False
    if mode == "exact":
        return la == lb and a.get("span") == b.get("span")
    return overlaps(la, lb)


def support_entry(pre, t):
    return {"file": pre.path, "model": pre.model, "span": t.get("span"),
            "span_start": t.get("span_start"), "span_end": t.get("span_end"),
            "harmfulness": t.get("harmfulness"), "harm_types": t.get("harm_types"),
            "error_source": t.get("error_source")}


def combine_gold(entries):
    """entries: [(Preselection, gold, detail)] -> (gold, detail, disagreement or None).

    The gold is the first non-null one; the disagreement lists the verdicts of all files that have one."""
    known = [(p, g, d) for p, g, d in entries if g is not None]
    if not known:
        return None, None, None
    _, gold, detail = known[0]
    if len({g for _, g, _ in known}) > 1 or len({d for _, _, d in known if d is not None}) > 1:
        return gold, detail, [{"file": p.path, "model": p.model, "gold_annotations": g,
                               "gold_annotation_details": d} for p, g, d in known]
    return gold, detail, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="outputs of 40_find_harmful_errors.py (two or more)")
    ap.add_argument("-o", "--output", required=True, help="output JSONL")
    ap.add_argument("--match", choices=("overlap", "exact", "segment"), default="overlap",
                    help="when two flagged spans agree (default: overlap)")
    ap.add_argument("--min-files", type=int, default=None,
                    help="keep targets flagged by at least this many files (default: all files)")
    args = ap.parse_args()

    pres = [Preselection(p) for p in args.files]
    n = len(pres)
    min_files = args.min_files or n
    if not 1 <= min_files <= n:
        ap.error(f"--min-files must be between 1 and {n}")
    if n == 1:
        shout("note: only one input file; the output is a copy of its flagged targets")

    out_records = {}                       # audio name -> output record (insertion order = output order)
    out_gold = {}                          # audio name -> {"gold", "details", "dis"}: lists parallel to targets
    kept_targets = defaultdict(list)       # (audio name, system) -> kept targets, to avoid duplicates
    flags = [0] * n
    kept_from = [0] * n
    pair = [[0] * n for _ in range(n)]     # pair[i][j]: flags of file i supported by file j
    intersected_from = [{"file": p.path, "model": p.model} for p in pres]

    for i, pre in enumerate(pres):
        for rec in pre.records:
            name = audio_name(rec.get("audio"))
            for t in rec.get("targets", []):
                if t.get("span") is None:
                    continue
                key = (name, t.get("system"))
                flags[i] += 1
                agree = [(i, t)]
                for j, other in enumerate(pres):
                    if j == i:
                        continue
                    m = next((u for u in other.index.get(key, []) if match(t, u, args.match)), None)
                    if m is not None:
                        agree.append((j, m))
                        pair[i][j] += 1
                if len(agree) < min_files:
                    continue
                if any(match(t, k, args.match) for k in kept_targets[key]):
                    continue  # already kept through a file earlier in the list
                kept = copy.deepcopy(t)
                kept["n_agree"] = len(agree)
                kept["agreed_by"] = [support_entry(pres[j], u) for j, u in sorted(agree)]
                gold, detail, dis = combine_gold(
                    [(pres[j], *pres[j].gold_of.get(id(u), (None, None))) for j, u in sorted(agree)])
                kept_targets[key].append(t)
                if name not in out_records:
                    out = copy.deepcopy(rec)
                    out["targets"] = []
                    for stale in ("gold_annotations", "gold_annotation_details", "gold_disagreement"):
                        out.pop(stale, None)  # rebuilt below for the output targets
                    out["intersected_from"] = intersected_from
                    out_records[name] = out
                    out_gold[name] = {"gold": [], "details": [], "dis": []}
                out_records[name]["targets"].append(kept)
                for key_, value in (("gold", gold), ("details", detail), ("dis", dis)):
                    out_gold[name][key_].append(value)
                kept_from[i] += 1

    asr_added, asr_missing = 0, 0
    for name, rec in out_records.items():
        system = rec.get("asr_system")
        if not system:
            asr_missing += 1
            continue
        if not any(t.get("system") == system for t in rec["targets"]):
            rec["targets"].append({"tgt_lan": rec.get("src_language"), "system": system,
                                   "text": rec.get("asr"), "span": None, "span_start": None,
                                   "span_end": None})
            gold, detail, dis = combine_gold([(p, *p.system_gold[(name, system)]) for p in pres
                                              if (name, system) in p.system_gold])
            for key_, value in (("gold", gold), ("details", detail), ("dis", dis)):
                out_gold[name][key_].append(value)
            asr_added += 1

    has_gold = any(p.has_gold for p in pres)
    n_disagree = n_null = 0
    if has_gold:
        for name, rec in out_records.items():
            g = out_gold[name]
            rec["gold_annotations"], rec["gold_annotation_details"] = g["gold"], g["details"]
            rec["gold_disagreement"] = g["dis"]
            n_disagree += sum(1 for x in g["dis"] if x is not None)
            n_null += sum(1 for x in g["gold"] if x is None)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in out_records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(kept_from)
    if asr_missing:
        shout(f"WARNING: {asr_missing} output segments have no asr_system; the ASR target could not be added")
    shout(f"-> {args.output}: {len(out_records)} segments, {total} flagged targets "
          f"(match={args.match}, kept if flagged by >= {min_files} of {n} files); "
          f"ASR target without a span added to {asr_added} segments")
    for i, pre in enumerate(pres):
        label = f"{pre.model} " if pre.model else ""
        shout(f"  [{i}] {label}{pre.path}: {len(pre.records)} segments, {flags[i]} flagged targets, "
              f"{kept_from[i]} kept from this file")
        if pre.bad_gold:
            shout(f"      WARNING: {pre.bad_gold} records have gold_annotations of another length than "
                  f"targets; their gold is ignored")
    if has_gold:
        n_targets = sum(len(r["targets"]) for r in out_records.values())
        shout(f"  gold: kept for {n_targets - n_null} of {n_targets} output targets; "
              f"{n_disagree} targets with gold_disagreement; {n_null} targets without any gold")
    if n > 1:
        shout("  agreement: share of the flags of the row file that the column file also flagged")
        shout("      " + "".join(f"{'[' + str(j) + ']':>8}" for j in range(n)))
        for i in range(n):
            cells = ["       -" if i == j else
                     f"{pair[i][j] / flags[i] * 100 if flags[i] else 0:>7.0f}%" for j in range(n)]
            shout(f"  [{i}] " + "".join(cells) + f"   ({flags[i]} flags)")


if __name__ == "__main__":
    main()