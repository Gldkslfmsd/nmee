#!/usr/bin/env python3
"""Find harmful errors with an LLM in the aligned, sentence-level JSONL.

Input: output of 30_add_and_align_sentences.py -- one sentence per line with several systems
(canary_asr, canary_cs, ..., gold_transcript) in "text".

Output: one line per segment in which at least one harmful error was found:

  {"document", "dataset", "src_language", "audio", "beg", "end", "segmented_by",   # identification
   "asr": "...", "asr_system": "canary_asr",
   "gold_transcript": "...",                                                        # if available
   "targets": [{"tgt_lan": "cs", "system": "canary_cs", "text": "<the whole segment>",
                "reference": "<human reference, if any>",
                "span": "...", "span_start": 22, "span_end": 41,
                "intended": "...", "harm_types": ["False attribution"],
                "harmfulness": 4, "explanation": "...", "error_source": "MT"}],
   "annotator": {"model": ..., "backend": ..., "level": "segment", "shown": [...]}}

What the LLM sees and what it annotates is chosen with --show and --annotate; both take system names,
or "all", "targets" (machine outputs in another language than the source) and "human". Examples:

    # errors in the transcript and in the Czech translation, with both shown
    --show canary_asr canary_cs --annotate canary_asr canary_cs
    # everything shown (gold transcript included), errors looked for in all machine outputs
    --show all --annotate canary_asr canary_cs canary_de canary_it canary_sk
    # gold transcript as the source of truth, only the translations annotated
    --show gold_transcript canary_asr targets --annotate targets

By default --annotate is every machine system and --show is the gold transcript, the transcript, the
annotated systems and any human reference. A human reference in the target language, if present in the
input as a human system, is shown to the model and copied into "reference".

--domain adds a static description of the situation ("This is an earnings call of a US company.",
"This is the Czech Parliament."). --context N also shows N neighbouring sentences (not annotated).
--level document is not implemented yet.

Usage:
    python 40_find_harmful_errors.py --input aligned/all.jsonl --output harmful.jsonl \\
        --backend vllm --model Qwen/Qwen3-4B-Instruct-2507 \\
        --show all --annotate canary_asr canary_cs --domain "This is an earnings call."
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import harm_llm
import llm_backend
import prompts


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="output of 30_add_and_align_sentences.py")
    ap.add_argument("--output", required=True, help="harmful errors, one segment per line")
    ap.add_argument("--show", nargs="+", help="systems shown to the LLM (names, all, targets, human)")
    ap.add_argument("--annotate", nargs="+", help="systems to look for errors in (default: all machine "
                                                  "systems)")
    ap.add_argument("--asr-system", help="system used as the transcript (default: the first machine "
                                         "system in the source language)")
    ap.add_argument("--gold-system", help="human transcript (default: the first human system in the "
                                          "source language)")
    ap.add_argument("--strict-systems", action="store_true",
                    help="fail if a name in --show/--annotate is missing from a line")
    ap.add_argument("--level", choices=["segment", "document"], default="segment",
                    help="unit of work (document level is not implemented yet)")
    ap.add_argument("--context", type=int, default=0,
                    help="also show N neighbouring sentences of the same document (not annotated)")
    ap.add_argument("--domain", help="static description of the situation, added to every prompt")
    ap.add_argument("--domain-file", help="read --domain from a file")
    ap.add_argument("--limit", type=int, help="only the first N segments (for testing)")
    ap.add_argument("--keep-raw", action="store_true", help="store the raw LLM answer in the output")
    ap.add_argument("--keep-gold-meta", action="store_true", help="copy \"gold_meta\" to the output")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    llm_backend.add_arguments(ap)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.level != "segment":
        sys.exit("--level document is not implemented yet")
    if args.domain_file:
        args.domain = Path(args.domain_file).read_text(encoding="utf-8").strip()

    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if not recs:
        sys.exit(f"{args.input} is empty")
    if args.limit:
        recs = recs[:args.limit]
    work = harm_llm.build_views(recs, args)
    if not work:
        sys.exit("nothing to annotate: --annotate matched no system with text")
    systems = sorted({s for _, v in work for s in v["annotate"]})
    print(f"{len(recs)} segments, {len(work)} to annotate, systems: {', '.join(systems)}",
          file=sys.stderr)

    chats = [prompts.build_chat(v) for _, v in work]
    if args.dry_run:
        print(chats[0][0]["content"] + "\n\n" + chats[0][1]["content"])
        return

    backend = llm_backend.get_backend(llm_backend.BackendConfig.from_args(args))
    t0 = time.time()
    answers = backend.generate(chats, prompts.response_schema(systems))
    print(f"generated {len(answers)} answers in {time.time() - t0:.0f}s", file=sys.stderr)

    stats = Counter()
    n_out = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for (ctx, view), answer in zip(work, answers):
            data = harm_llm.extract_json(answer)
            if data is None:
                stats["unparsable answer"] += 1
                continue
            ctx["shown"] = view["shown"]
            entries = [e for e in (harm_llm.clean_annotation(a, ctx, stats)
                                   for a in data["annotations"]) if e]
            for e in entries:
                stats[f"error in {e['system']}"] += 1
            rec = harm_llm.build_output(ctx, entries, args, raw=answer)
            if rec:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"-> {args.output}: {n_out} segment(s) with harmful errors "
          f"({sum(v for k, v in stats.items() if k.startswith('error in'))} error(s))", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
