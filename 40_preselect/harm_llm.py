#!/usr/bin/env python3
"""Turning the aligned JSONL into LLM work and the LLM answers into harmful-error records.

Input: the output of 30_add_and_align_sentences.py (one sentence per line, several systems per line).
Output: one line per segment in which at least one harmful error was found, in the harmful-error format
(see find_harmful_errors.py --help).

The unit of work is a segment by default; document level is prepared for but not implemented yet
(build_views() is the only place that would change).
"""
import difflib
import json
import re
import sys

import prompts

IDENT_FIELDS = ("document", "dataset", "src_language", "audio", "beg", "end", "segmented_by")


# ---------------------------------------------------------------- choosing the systems

def resolve_systems(rec, args):
    """(asr_system, gold_system, references{lan: system}, shown[...], annotate[...]) for one record."""
    info = rec["systems_info"]
    src_lan = rec.get("src_language")

    asr = args.asr_system
    if asr is None:
        asr = next((s for s, v in info.items()
                    if not v.get("is_human") and (src_lan is None or v.get("lan") == src_lan)), None)
    if src_lan is None:  # no "src_language" in the file: take the transcript's language
        src_lan = info.get(asr, {}).get("lan")
    gold = args.gold_system
    if gold is None:
        gold = next((s for s, v in info.items()
                     if v.get("is_human") and (src_lan is None or v.get("lan") == src_lan)), None)
    references = {v["lan"]: s for s, v in info.items()
                  if v.get("is_human") and s != gold and v.get("lan") != src_lan}

    def expand(names, default):
        if not names:
            return [s for s in default if s in info]
        out = []
        for n in names:
            if n == "all":
                out += list(info)
            elif n == "targets":  # machine outputs in another language than the source
                out += [s for s, v in info.items()
                        if not v.get("is_human") and v.get("lan") != src_lan]
            elif n == "human":
                out += [s for s, v in info.items() if v.get("is_human")]
            elif n in info:
                out.append(n)
            elif args.strict_systems:
                sys.exit(f"{n!r} is not a system of this file (have: {', '.join(info)})")
            else:
                continue
            out = list(dict.fromkeys(out))
        return out

    annotate = expand(args.annotate, [s for s, v in info.items() if not v.get("is_human")])
    shown = expand(args.show, [x for x in ([gold, asr] if gold else [asr]) if x] + annotate
                   + list(references.values()))
    shown = [s for s in dict.fromkeys(shown) if rec["text"].get(s, "").strip()]
    annotate = [s for s in annotate if s in shown]
    return asr, gold, references, shown, annotate


def role_of(system, asr, gold, references):
    if system == gold:
        return "gold"
    if system in references.values():
        return "reference"
    if system == asr:
        return "source"
    return "target"


# ---------------------------------------------------------------- views (units of work)

def build_views(recs, args):
    """[(records, view)] -- one per segment for now."""
    views = []
    for i, rec in enumerate(recs):
        asr, gold, references, shown, annotate = resolve_systems(rec, args)
        if not annotate:
            continue
        info = rec["systems_info"]
        view = {
            "domain": args.domain,
            "shown": [{"system": s, "lan": info[s].get("lan"), "text": rec["text"][s],
                       "role": role_of(s, asr, gold, references)} for s in shown],
            "annotate": annotate,
        }
        if args.context > 0:
            src = gold or asr
            before = [recs[j]["text"].get(src, "") for j in range(max(0, i - args.context), i)
                      if recs[j].get("document") == rec.get("document")]
            after = [recs[j]["text"].get(src, "")
                     for j in range(i + 1, min(len(recs), i + 1 + args.context))
                     if recs[j].get("document") == rec.get("document")]
            view["context_before"] = [t for t in before if t]
            view["context_after"] = [t for t in after if t]
        views.append(({"record": rec, "asr": asr, "gold": gold, "references": references,
                       "annotate": annotate}, view))
    return views


# ---------------------------------------------------------------- parsing the answers

def extract_json(response):
    start, end = response.find("{"), response.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(response[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data.get("annotations"), list) else None


def locate(span, text):
    """(start, end) of span in text: exact, then case-insensitive, then fuzzy."""
    i = text.find(span)
    if i >= 0:
        return i, i + len(span)
    i = text.lower().find(span.lower())
    if i >= 0:
        return i, i + len(span)
    m = difflib.SequenceMatcher(None, text.lower(), span.lower()).find_longest_match(
        0, len(text), 0, len(span))
    if m.size >= max(4, 0.6 * len(span)):
        return m.a, m.a + m.size
    return None


def clean_annotation(a, ctx, stats):
    """Validate one LLM annotation against the segment; returns a target entry or None."""
    rec = ctx["record"]
    system = a.get("system")
    if system not in ctx["annotate"]:
        stats["wrong system"] += 1
        return None
    text = rec["text"].get(system, "")
    span = str(a.get("span", "")).strip()
    if not span:
        stats["empty span"] += 1
        return None
    pos = locate(span, text)
    if pos is None:
        stats["span not found"] += 1
        return None
    start, end = pos
    harm = [h for h in a.get("harm_types", []) if h in prompts.HARM_TYPES]
    if not harm:
        harm = ["Other"]
    try:
        harmfulness = int(a.get("harmfulness"))
    except (TypeError, ValueError):
        harmfulness = None
    if harmfulness is not None:
        harmfulness = min(5, max(1, harmfulness))
    lan = rec["systems_info"][system].get("lan")
    entry = {
        "tgt_lan": lan,
        "system": system,
        "text": text,
        "reference": rec["text"].get(ctx["references"].get(lan)) if lan in ctx["references"] else None,
        "span": text[start:end],
        "span_start": start,
        "span_end": end,
        "intended": str(a.get("intended", "")).strip() or None,
        "harm_types": harm,
        "harmfulness": harmfulness,
        "explanation": str(a.get("explanation", "")).strip() or None,
        "error_source": a.get("error_source") if a.get("error_source") in prompts.ERROR_SOURCES else None,
    }
    return {k: v for k, v in entry.items() if v is not None}


def build_output(ctx, annotations, args, raw=None):
    """One output record for a segment with at least one accepted error, or None."""
    if not annotations:
        return None
    rec = ctx["record"]
    out = {k: rec[k] for k in IDENT_FIELDS if k in rec}
    if ctx["asr"]:
        out["asr"] = rec["text"].get(ctx["asr"])
        out["asr_system"] = ctx["asr"]
    if ctx["gold"]:
        out["gold_transcript"] = rec["text"].get(ctx["gold"])
    if args.keep_gold_meta and rec.get("gold_meta"):
        out["gold_meta"] = rec["gold_meta"]
    out["targets"] = annotations
    out["annotator"] = {"model": args.model, "backend": args.backend, "level": args.level,
                        # which systems the LLM was asked about: without this, a system missing from
                        # "targets" could mean "no error found" or "never looked at"
                        "annotated": list(ctx["annotate"]),
                        "shown": [s["system"] for s in ctx.get("shown", [])] or None}
    out["annotator"] = {k: v for k, v in out["annotator"].items() if v is not None}
    if raw is not None and args.keep_raw:
        out["llm_raw"] = raw
    return out


def normalise_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()
