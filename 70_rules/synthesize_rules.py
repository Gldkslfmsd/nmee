#!/usr/bin/env python3
"""Synthesize verification rules for confirmed harmful spans with an LLM.

For every confirmed harmful span in the data (the same JSONL as for rule_editor.py), the LLM gets
the source, the reference translation with the span marked as >>span<<, and is asked for
--count verification rules that check exactly that error. The rules are printed as JSONL, one rule
per line, in the format that rule_editor.py --rules reads:

  {"key": "<seg>|<system>|<span index>", "rule_id": "<seg>.rN", "canary_span": ["<span>"],
   "rule": "...", "model": "..."}

Usage:
  python synthesize_rules.py confirmed.jsonl --backend einfra --model gpt-oss-120B > rules.jsonl
  python rule_editor.py confirmed.jsonl --systems systems.jsonl --rules rules.jsonl -o rules_verified.jsonl

The backend options (--backend, --model, --api-base, ...) come from llm_backend.py.
"""
import argparse
import json
import re
import sys
import time

import llm_backend
from llm_verifier import add_thinking_argument, apply_extra_body, safe_generate_iter, thinking_body
from rule_editor import locate, seg_id

PROMPT = """Given this text from {source_lang} to {target_lang}, generate {verification_rules_count} verification rules that
can be used to check the quality of translations. The rule(s) should be concise, clear, and specific. Avoid
vague or generic rules. Each rule should be a single sentence and should be evaluatable as true or false given
a translation. An example of rules:
[
“The word for ‘paper’ should be ‘Artikel’ or English loanword ‘Paper’ but not ‘Papier’.”,
“In this context, ‘撒娇’ should be equivalent to acting cute and adorable, instead of having any flirtatious
or sexual sense.”,
“The translations for both the slaps should be different, but close enough lexically.”
]
Output the rules in JSON format as above.

Source text ({source_lang}):
{source_text}

Translation ({target_lang}):
{translation}

The translation contains a harmful error in the part marked with >> and <<: >>{span}<<.
The rules must check this error only: a translation that makes this error, or another error in
the same place, must fail at least one rule, and a correct translation must pass all of them.
The rules will be checked against other translations of the same source text, so do not refer to
the markers or to this particular translation."""

LANGUAGES = {"en": "English", "cs": "Czech", "de": "German", "fr": "French", "es": "Spanish",
             "it": "Italian", "pl": "Polish", "sk": "Slovak", "uk": "Ukrainian", "ru": "Russian",
             "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "nl": "Dutch",
             "pt": "Portuguese", "hu": "Hungarian", "ro": "Romanian", "tr": "Turkish"}


def lang_name(code):
    return LANGUAGES.get((code or "").split("-")[0].lower(), code or "the target language")


def fill(template, **values):
    """Fill {name} placeholders in one pass (braces inside the texts are left alone)."""
    return re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
                  template)


def mark(text, start, end, span):
    """The text with the span marked as >>span<<."""
    if start is None:
        return text
    return text[:start] + ">>" + text[start:end] + "<<" + text[end:]


def parse_rules(answer):
    """A list of rule strings from the LLM answer (a JSON list, possibly with curly quotes,
    code fences or a thinking block around it); None if no list can be found."""
    if answer is None:
        return None
    t = re.sub(r"<think>.*?</think>", "", answer, flags=re.S)
    t = re.sub(r"```(?:json)?", "", t)
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        block = t[i:j + 1]
        for cand in (block, re.sub(r"[“”]", '"', block)):
            try:
                v = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, dict) and isinstance(v.get("rules"), list):
                return [str(x).strip() for x in v["rules"] if str(x).strip()]
        # a list of quoted strings that is not valid JSON
        found = re.findall(r"[\"“]((?:[^\"“”\\]|\\.)+)[\"”]\s*,?", block)
        if found:
            return [f.strip() for f in found if f.strip()]
    return None


def load_spans(path):
    """One entry per confirmed harmful span, with the same key as in rule_editor.py."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sid = seg_id(d)
            for t in d.get("targets", []):
                for i, sp in enumerate(t.get("harm_spans", [])):
                    s, e = locate(t["text"], sp)
                    if s is None:
                        print(f"WARNING: {sid}: span {sp['span']!r} not found in the {t['system']} text; "
                              f"the translation is shown without markers", file=sys.stderr)
                    out.append({"key": f"{sid}|{t['system']}|{i}", "seg": sid, "span": sp["span"],
                                "source": d.get("gold_transcript", ""),
                                "src_lan": d.get("src_language"), "tgt_lan": t.get("tgt_lan"),
                                "marked": mark(t["text"], s, e, sp["span"])})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="confirmed harmful spans, JSONL (targets with harm_spans)")
    p.add_argument("--count", "-n", type=int, default=2, help="how many rules per span (default 2)")
    p.add_argument("--output", "-o", help="write the rules here instead of printing them")
    p.add_argument("--prompt", help="prompt file with {source_lang}, {target_lang}, "
                                    "{verification_rules_count}, {source_text}, {translation}, {span}")
    p.add_argument("--print-prompt", action="store_true", help="print the prompt of the first span and exit")
    add_thinking_argument(p)
    llm_backend.add_arguments(p)
    args = p.parse_args()

    template = PROMPT
    if args.prompt:
        with open(args.prompt, encoding="utf-8") as f:
            template = f.read()
    spans = load_spans(args.data)
    if not spans:
        sys.exit(f"no confirmed harm_spans in {args.data}")
    chats = [[{"role": "user", "content": fill(
        template, source_lang=lang_name(s["src_lan"]), target_lang=lang_name(s["tgt_lan"]),
        verification_rules_count=args.count, source_text=s["source"], translation=s["marked"],
        span=s["span"])}] for s in spans]
    if args.print_prompt:
        print(chats[0][0]["content"])
        return

    cfg = llm_backend.BackendConfig.from_args(args)
    model = f"{cfg.backend}:{cfg.model}"
    if cfg.backend == "dummy":   # testing without a model
        answers = (json.dumps([f"The part '{s['span']}' must be translated correctly (rule {k + 1})."
                               for k in range(args.count)]) for s in spans)
    else:
        backend = llm_backend.get_backend(cfg)
        apply_extra_body(backend, thinking_body(args.enable_thinking))
        answers = safe_generate_iter(backend, chats, None)

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    t0, n_rules, failed, per_seg = time.time(), 0, [], {}
    try:
        for n, (s, a) in enumerate(zip(spans, answers), 1):
            rules = parse_rules(a)
            if not rules:
                failed.append(s["key"])
                print(f"WARNING: {s['key']}: no rules in the LLM answer: "
                      f"{'request failed' if a is None else a[:500]!r}", file=sys.stderr)
                continue
            if len(rules) != args.count:
                print(f"note: {s['key']}: {len(rules)} rules instead of {args.count}", file=sys.stderr)
            for r in rules:
                per_seg[s["seg"]] = per_seg.get(s["seg"], 0) + 1   # rule ids unique per segment
                out.write(json.dumps({"key": s["key"], "rule_id": f"{s['seg']}.r{per_seg[s['seg']]}",
                                      "canary_span": [s["span"]], "rule": r, "model": model},
                                     ensure_ascii=False) + "\n")
                n_rules += 1
            out.flush()
            if n % 20 == 0 or n == len(spans):
                print(f"  {n}/{len(spans)} spans ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    finally:
        if args.output:
            out.close()
    print(f"{n_rules} rules for {len(spans) - len(failed)} of {len(spans)} spans"
          + (f"; no rules for {len(failed)}: {', '.join(failed[:10])}" + (" ..." if len(failed) > 10 else "")
             if failed else ""), file=sys.stderr)


if __name__ == "__main__":
    main()