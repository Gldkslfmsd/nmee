#!/usr/bin/env python3
"""Do other systems make the same harmful error that a human annotator confirmed in one system?

    python3 same_harm.py --confirmed confirmed.fr.jsonl v1.earnings.fr.jsonl -o same_harm.fr.jsonl \\
        --backend einfra --model Kimi
    python3 same_harm.py --confirmed confirmed.fr.jsonl v1.earnings.fr.jsonl -o x.jsonl --dry-run

--confirmed: output of only_confirmed.py (can be repeated): segments with human-confirmed harm spans.
input:       segments with the outputs of other systems, in the format of 30_add_and_align_sentences.py
             ("systems_info", "text": {system: text}), e.g. v1.earnings.fr.jsonl, or the output of
             filter_harm.py of it. The input is filtered with filter_harm.match() first, so both work.

For every confirmed harm span and every other machine system in the same language (default; or the
systems given with --systems), the LLM gets one short request with the correct transcript, the output
with the confirmed span marked >>like this<<, and the other system's output, and answers yes or no:
does the other output contain the same harmful error? Each request is about one segment only.

Output: JSONL, one line per confirmed segment, in the order of the confirmed file(s): the confirmed
record, where every item of "harm_spans" gets
    "same_harm": {"<other system>": true | false | null, ...}
        true = the same error, false = not, null = the request failed or the answer was unparsable
    "same_harm_raw": {"<other system>": "<the LLM answer>"}         (with --keep-raw only)
and the segment gets
    "other_outputs": {"<other system>": "<its text>", ...}
    "same_harm_judge": {"model": ..., "backend": ...}
Records are written as soon as all their requests are answered. Run statistics go to stderr; at the
end, the statistics per system (eval_systems.py) are printed to stdout.

Extension point: build_chat() takes context_before / context_after (lists of neighbouring segments),
not used yet. When the automatic alignment of the other systems is wrong, the error may be in a
neighbouring segment. For that, the neighbours have to come from the full input (v1.*.jsonl), not
from a filter_harm.py output, which has only the confirmed segments.
"""
import argparse
import copy
import json
import re
import sys
import time
from collections import Counter

import eval_systems
import filter_harm
import llm_backend

OPEN, CLOSE = ">>", "<<"

SYSTEM_MESSAGE = "You compare outputs of speech translation systems. You answer only with valid JSON."

TASK = f"""A speech was transcribed and translated automatically. A human annotator found a harmful error in the output of system A; the erroneous part is marked {OPEN}like this{CLOSE}.

Does the output of system B contain the same error, i.e. the same wrong meaning at this place, possibly in different words? Answer "no" if system B is correct at this place or makes a different error.

Answer with {{"answer": "yes"}} or {{"answer": "no"}} and nothing else."""

SCHEMA = {"type": "object",
          "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
          "required": ["answer"], "additionalProperties": False}

PROGRESS_SECONDS = 30.0


def shout(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- prompt

def mark(text, start, end):
    return text[:start] + OPEN + text[start:end] + CLOSE + text[end:]


def build_chat(job, context_before=None, context_after=None):
    """The chat for one (segment, confirmed span, other system).
    context_before / context_after: neighbouring segments (reserved for later, not used yet)."""
    lines = []
    if job["transcript"]:
        lines.append(f"Correct transcript ({job['src_lan']}): {job['transcript']}")
    else:
        lines.append(f"Automatic transcript ({job['src_lan']}, may contain errors): {job['asr']}")
    lines.append(f"System A ({job['lan_a']}): {job['marked_a']}")
    lines.append(f"System B ({job['lan_b']}): {job['text_b']}")
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": TASK + "\n\n" + "\n".join(lines)}]


def parse_answer(response):
    """True / False / None."""
    if response is None:
        return None
    m = re.search(r'"answer"\s*:\s*"(yes|no)"', response, re.I)
    if m:
        return m.group(1).lower() == "yes"
    words = re.findall(r"\b(yes|no)\b", response, re.I)  # last one: reasoning may come first
    return words[-1].lower() == "yes" if words else None


# ---------------------------------------------------------------- work

def other_systems(inp, target, args):
    """Machine systems of the input record in the language of the target, without the target's own."""
    info, texts = inp.get("systems_info") or {}, inp.get("text") or {}
    lan = (target.get("tgt_lan") or "").lower()
    out = []
    for s, v in info.items():
        if s == target.get("system") or not (texts.get(s) or "").strip():
            continue
        if args.systems:
            if s in args.systems:
                out.append(s)
        elif not v.get("is_human") and (v.get("lan") or "").lower() == lan:
            out.append(s)
    return out


def human_transcript(inp, src_lan):
    for s, v in (inp.get("systems_info") or {}).items():
        if v.get("is_human") and v.get("lan") == src_lan:
            return (inp.get("text") or {}).get(s)
    return None


def build_segments(matched, args, stats):
    """[{"record": output record, "jobs": [job, ...]}] in the order of the confirmed file(s)."""
    segments = []
    for name, conf_recs, items in matched:
        if len(items) > 1:
            shout(f"WARNING: {name} is {len(items)}x in the input; using the first")
            stats["segments more than once in the input"] += 1
        inp = items[0][1]
        out = copy.deepcopy(conf_recs[0])
        out["targets"] = [copy.deepcopy(t) for r in conf_recs for t in r.get("targets", [])]
        src_lan = out.get("src_language") or inp.get("src_language")
        transcript = out.get("gold_transcript") or human_transcript(inp, src_lan)
        others, jobs = {}, []
        for t in out["targets"]:
            cands = other_systems(inp, t, args)
            for sp in t.get("harm_spans", []):
                s, e, text = sp["span_start"], sp["span_end"], t.get("text") or ""
                if text[s:e] != sp.get("span"):
                    shout(f"WARNING: {name} {t.get('system')}: text[{s}:{e}] = {text[s:e]!r}, "
                          f"but the span is {sp.get('span')!r}")
                    stats["spans not matching the text"] += 1
                sp["same_harm"] = {c: None for c in cands}
                for c in cands:
                    others[c] = inp["text"][c]
                    jobs.append({"span": sp, "system_b": c, "src_lan": src_lan,
                                 "transcript": transcript, "asr": out.get("asr"),
                                 "lan_a": t.get("tgt_lan"), "marked_a": mark(text, s, e),
                                 "lan_b": (inp["systems_info"][c].get("lan") or t.get("tgt_lan")),
                                 "text_b": inp["text"][c]})
        out["other_outputs"] = others
        out["same_harm_judge"] = {"model": args.model, "backend": args.backend}
        if not jobs:
            stats["segments without other systems to compare"] += 1
        segments.append({"record": out, "jobs": jobs})
    return segments


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmed", action="append", required=True, metavar="FILE",
                    help="output of only_confirmed.py (can be repeated)")
    ap.add_argument("input", help="segments with the other systems (v1.*.jsonl or filter_harm.py output)")
    ap.add_argument("-o", "--output", required=True, help="output JSONL")
    ap.add_argument("--systems", nargs="+",
                    help="other systems to compare (default: all machine systems in the language of "
                         "the confirmed output)")
    ap.add_argument("--limit", type=int, help="only the first N confirmed segments (for testing)")
    ap.add_argument("--keep-raw", action="store_true", help="store the raw LLM answers")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    llm_backend.add_arguments(ap)
    args = ap.parse_args()

    stats = Counter()
    matched = filter_harm.match(args.confirmed, args.input, stats)
    if args.limit:
        matched = matched[:args.limit]
    segments = build_segments(matched, args, stats)
    jobs = [j for seg in segments for j in seg["jobs"]]
    shout(f"{len(segments)} confirmed segments, {len(jobs)} comparisons")
    if args.dry_run:
        if jobs:
            chat = build_chat(jobs[0])
            print(chat[0]["content"] + "\n\n" + chat[1]["content"])
        return
    if not jobs:
        shout("nothing to compare")

    backend = llm_backend.get_backend(llm_backend.BackendConfig.from_args(args)) if jobs else None
    chats = [build_chat(j) for j in jobs]
    remaining = [len(seg["jobs"]) for seg in segments]
    owner = [i for i, seg in enumerate(segments) for _ in seg["jobs"]]
    next_seg, done, t0, last_log = 0, 0, time.time(), time.time()

    with open(args.output, "w", encoding="utf-8", buffering=1) as out:
        def flush_ready():
            nonlocal next_seg
            while next_seg < len(segments) and remaining[next_seg] == 0:
                out.write(json.dumps(segments[next_seg]["record"], ensure_ascii=False) + "\n")
                next_seg += 1

        flush_ready()  # segments without anything to compare
        answers = backend.generate_iter(chats, SCHEMA) if jobs else []
        for k, (job, response) in enumerate(zip(jobs, answers)):
            verdict = parse_answer(response)
            job["span"]["same_harm"][job["system_b"]] = verdict
            if args.keep_raw:
                job["span"].setdefault("same_harm_raw", {})[job["system_b"]] = response
            if response is None:
                stats["failed requests"] += 1
            elif verdict is None:
                stats["unparsable answers"] += 1
            else:
                stats["answers yes" if verdict else "answers no"] += 1
            remaining[owner[k]] -= 1
            flush_ready()
            done += 1
            if time.time() - last_log >= PROGRESS_SECONDS or done == len(jobs):
                last_log = time.time()
                shout(f"[{time.strftime('%H:%M:%S')}] {done}/{len(jobs)} comparisons, "
                      f"{next_seg}/{len(segments)} segments written, {last_log - t0:.0f}s")

    shout(f"-> {args.output}")
    for k, v in stats.items():
        shout(f"  {k}: {v}")
    # statistics per system of the written segments (the same as eval_systems.py on the output)
    print()
    eval_systems.print_table(*eval_systems.collect(seg["record"] for seg in segments[:next_seg]))


if __name__ == "__main__":
    main()