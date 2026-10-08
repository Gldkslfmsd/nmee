#!/usr/bin/env python3
"""Add the human annotations from pearmut to the LLM preselection (output of 40_find_harmful_errors.py).

    python3 pearmut_to_jsonl.py --pearmut ann1.jsonl ann2.jsonl --input out/Kimi_cs.jsonl \\
        -o gold/Kimi_cs.gold.jsonl
    # one output file per annotator:
    python3 pearmut_to_jsonl.py --pearmut ann*.jsonl --input out/Kimi_cs.jsonl -o 'gold/Kimi_cs.{user}.jsonl'

--input: output of 40_find_harmful_errors.py (one line per segment, one or more flagged "targets").
--pearmut: pearmut annotation JSONL (one line per page: "item", "annotation", "user_id"); error spans are
  character offsets with an inclusive end. Two special tokens follow the text of every system output:
  [no harm] at offset len(text) and [undecidable] at offset len(text) + 1. A span starting at one of these
  offsets is not an error: it is the annotator's verdict on the whole output of that system.

Output: the input records of the segments annotated in pearmut, plus
  "gold_annotations": [true, false, null, ...]   -- one per item of "targets"; the verdict is about the
  whole output of the target's system in this segment, wherever the error is (not only at the flagged span):
    true   the annotator marked at least one error span in this system's output
    false  no error span, and the annotator selected [no harm] or [undecidable] for it
    null   the annotator did not see this system in this segment, or marked nothing at all for it
  "gold_annotation_details": ["harmful", "no_harm", "undecidable", null, ...]   -- one per item of "targets":
    harmful       the annotator marked an error span in this system's output (gold_annotations true)
    undecidable   the annotator selected [undecidable] for this system's output (gold_annotations false)
    no_harm       the annotator selected [no harm] for this system's output (gold_annotations false)
    null          as in gold_annotations
  gold_annotations is not changed by this: it is false for both "no_harm" and "undecidable".
A system shown in pearmut that has no target in the record (e.g. the ASR, when the LLM flagged only
the translation) is added to "targets" as a target without a span:
    {"tgt_lan": ..., "system": ..., "text": ..., "span": null, "span_start": null, "span_end": null}
  and its gold annotation is decided in the same way.

Several --input files are merged; a segment and target system present in more than one file is reported
and kept from the first file only. Several --pearmut files are merged too. With {user} in -o there is one
output file per annotator; otherwise the labels of all annotators are combined: true only if every
annotator who saw the target marked it. In the details, "harmful" needs all annotators who saw the target,
"undecidable" is given if any of them chose it (and not all said harmful), else "no_harm".
Disagreements between annotators are reported either way.
The evaluation (confusion matrix, accuracy, precision, recall) is done separately by eval.py.
Segments are matched by the audio file name. The position of the flagged span does not decide the gold
annotation; the statistics only report the human error spans that overlap no flagged span.
"""
import argparse
import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict


MARKERS = ("no_harm", "undecidable")  # the special tokens after each output, in this order


def shout(msg):
    print(msg, file=sys.stderr)


def audio_name(path):
    return os.path.basename(path or "")


# ---------------------------------------------------------------- pearmut

def load_pearmut(paths, stats):
    """{audio file name: {user: {"item_id", "texts": {system: text}, "spans": {system: [(s, e)]},
    "markers": {system: {"no_harm" and/or "undecidable"}}}}}"""
    gold = defaultdict(dict)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                page = json.loads(line)
                user = page.get("user_id") or "unknown"
                items, anns = page.get("item") or [], page.get("annotation") or []
                if len(items) != len(anns):
                    shout(f"WARNING: {path}:{n}: {len(items)} items but {len(anns)} annotations")
                for item, ann in zip(items, anns):
                    m = re.search(r'src="([^"]+)"', item.get("src") or "")
                    if not m:
                        stats["pearmut items without audio (skipped)"] += 1
                        continue
                    texts = item.get("tgt") or {}
                    spans, markers = {}, {}
                    for system, text in texts.items():
                        text = text or ""
                        spans[system] = []
                        markers[system] = set()
                        for sp in ((ann or {}).get(system) or {}).get("error_spans") or []:
                            si, ei = sp.get("start_i"), sp.get("end_i")
                            if si is None or ei is None:
                                continue
                            si, ei = min(si, ei), max(si, ei)
                            if si >= len(text):  # special token after the text: [no harm], [undecidable]
                                if si - len(text) >= len(MARKERS):
                                    stats["marker offsets beyond the two special tokens (ignored)"] += 1
                                for k, name in enumerate(MARKERS):
                                    if si <= len(text) + k <= ei:
                                        markers[system].add(name)
                                continue
                            spans[system].append((si, min(len(text), ei + 1)))
                    # languages from the header of the suggestion table: "canary_cs (CS)"
                    lans = {sys_: lan.lower() for sys_, lan in re.findall(
                        r"<th[^>]*>\s*([^<>()]+?)\s*\(([\w-]+)\)\s*</th>", item.get("instructions") or "")}
                    name = audio_name(m.group(1))
                    if user in gold[name]:
                        stats["pearmut items annotated again by the same user (last kept)"] += 1
                    gold[name][user] = {"item_id": item.get("item_id"), "texts": texts, "spans": spans,
                                        "markers": markers, "lans": lans}
                    stats["pearmut items"] += 1
    return gold


# ---------------------------------------------------------------- LLM preselection

def load_inputs(paths, stats):
    """Records of all input files; a (segment, system) seen in an earlier file is dropped."""
    records, owner = [], {}   # owner: (audio, system) -> file it was taken from
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                name = audio_name(rec.get("audio"))
                kept = []
                for t in rec.get("targets", []):
                    key = (name, t.get("system"))
                    if key in owner and owner[key] != path:
                        shout(f"DUPLICATE: {rec.get('document')} {rec.get('beg')}-{rec.get('end')} "
                              f"{t.get('system')} in {owner[key]} and {path}; keeping {owner[key]}")
                        stats["duplicate targets dropped"] += 1
                        continue
                    owner[key] = path
                    kept.append(t)
                if kept:
                    rec["targets"] = kept
                    records.append(rec)
    return records


def judge(target, human, stats):
    """True / False / None for one target (one system output in the segment) and one annotator.

    The verdict is about the whole output of the system, wherever the error is:
      True   the annotator marked at least one error span in this system's output
      False  no error span, and [no harm] or [undecidable] was selected
      None   the system was not shown to this annotator, or nothing at all was marked for it
    """
    system = target.get("system")
    if system not in human["texts"]:
        return None
    has_markers = bool(human["markers"].get(system))
    if human["spans"].get(system):
        if has_markers:
            stats["systems with an error span and a [no harm]/[undecidable] token (counted as harmful)"] += 1
        return True
    if has_markers:
        return False
    stats["systems shown with no error span and no [no harm]/[undecidable] token (null)"] += 1
    return None


def detail(target, human, gold):
    """"harmful" / "undecidable" / "no_harm" / None for one target; gold is judge()'s result."""
    if gold is None:
        return None
    if gold:
        return "harmful"
    return "undecidable" if "undecidable" in human["markers"].get(target.get("system"), ()) else "no_harm"


def add_unflagged_systems(rec, by_user, stats):
    """Append a target without a span for every system shown in pearmut that has no target."""
    present = {t.get("system") for t in rec["targets"]}
    for h in by_user.values():
        for system, text in h["texts"].items():
            if system in present:
                continue
            present.add(system)
            if system == rec.get("asr_system"):
                lan, text = rec.get("src_language"), rec.get("asr") or text
            else:
                lan = h["lans"].get(system)
            rec["targets"].append({"tgt_lan": lan, "system": system, "text": text,
                                   "span": None, "span_start": None, "span_end": None})
            stats["targets without a span added"] += 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pearmut", nargs="+", required=True, help="pearmut annotation JSONL")
    ap.add_argument("--input", nargs="+", required=True,
                    help="the annotated LLM preselection: output(s) of 40_find_harmful_errors.py")
    ap.add_argument("-o", "--output", required=True,
                    help="output JSONL; {user} in the name writes one file per annotator")
    args = ap.parse_args()

    stats = Counter()
    gold = load_pearmut(args.pearmut, stats)
    records = load_inputs(args.input, stats)
    users = sorted({u for by_user in gold.values() for u in by_user})

    # labels[record index][target index] = {user: True/False/None}; details: same with the detail strings
    labels, details, matched = {}, {}, set()
    for i, rec in enumerate(records):
        name = audio_name(rec.get("audio"))
        if name not in gold:
            stats["input segments not annotated in pearmut"] += 1
            continue
        matched.add(name)
        add_unflagged_systems(rec, gold[name], stats)
        labels[i] = [{u: judge(t, h, stats) for u, h in gold[name].items()} for t in rec["targets"]]
        details[i] = [{u: detail(t, gold[name][u], v) for u, v in per_user.items()}
                      for t, per_user in zip(rec["targets"], labels[i])]
        for t, per_user in zip(rec["targets"], labels[i]):
            vals = {u: v for u, v in per_user.items() if v is not None}
            if len(set(vals.values())) > 1:
                stats["targets with disagreeing annotators"] += 1
                shout(f"DISAGREEMENT: {rec.get('document')} {rec.get('beg')}-{rec.get('end')} "
                      f"{t.get('system')} \"{t.get('span')}\": "
                      + ", ".join(f"{u}={v}" for u, v in sorted(vals.items())))
    stats["pearmut items not in --input"] = len(set(gold) - matched)

    # human spans that no flagged target overlaps (errors the LLM did not flag)
    flagged = defaultdict(list)  # (audio, system) -> [(start, end)]
    for i in labels:
        rec = records[i]
        for t in rec["targets"]:
            if t.get("span") is not None:
                flagged[(audio_name(rec.get("audio")), t.get("system"))].append(
                    (t.get("span_start"), t.get("span_end")))
    for name in matched:
        for h in gold[name].values():
            for system, spans in h["spans"].items():
                for hs, he in spans:
                    if not any(s is not None and e is not None and hs < e and s < he
                               for s, e in flagged.get((name, system), [])):
                        stats["human spans not overlapping any flagged span"] += 1

    def write(path, value_of, detail_of):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        counts, kinds, n = Counter(), Counter(), 0
        with open(path, "w", encoding="utf-8") as out:
            for i, rec in enumerate(records):
                if i not in labels:
                    continue
                values = [value_of(per_user) for per_user in labels[i]]
                detail_values = [detail_of(per_user) for per_user in details[i]]
                if all(v is None for v in values):
                    continue  # the annotator(s) did not see any of the targets
                out_rec = copy.deepcopy(rec)
                out_rec["gold_annotations"] = values
                out_rec["gold_annotation_details"] = detail_values
                out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                n += 1
                counts.update("null" if v is None else str(v).lower() for v in values)
                kinds.update("null" if v is None else v for v in detail_values)
        shout(f"-> {path}: {n} segments, targets: "
              + ", ".join(f"{k} {counts[k]}" for k in ("true", "false", "null"))
              + " | details: " + ", ".join(f"{k} {kinds[k]}" for k in ("harmful", "no_harm", "undecidable", "null")))

    if "{user}" in args.output:
        for u in users:
            write(args.output.replace("{user}", u), lambda per_user, u=u: per_user.get(u),
                  lambda per_user, u=u: per_user.get(u))
    else:
        if len(users) > 1:
            shout(f"note: {len(users)} annotators combined: true only if all who saw a target marked it")

        def combined(per_user):
            vals = [v for v in per_user.values() if v is not None]
            return all(vals) if vals else None

        def combined_detail(per_user):
            vals = [v for v in per_user.values() if v is not None]
            if not vals:
                return None
            if all(v == "harmful" for v in vals):
                return "harmful"
            return "undecidable" if "undecidable" in vals else "no_harm"

        write(args.output, combined, combined_detail)

    shout(f"annotators: {', '.join(users) or '-'}")
    for k, v in sorted(stats.items()):
        shout(f"  {k}: {v}")


if __name__ == "__main__":
    main()