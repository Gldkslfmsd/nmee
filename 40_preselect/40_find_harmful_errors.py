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

The output is written gradually (every record is flushed as soon as its answer arrives, so you can
`tail -f` it). Progress is shown as a tqdm bar when stderr is a terminal and tqdm is installed;
otherwise (log files, batch jobs) it is logged as a line every few segments / seconds.

Grouping: by default every segment is a separate request. --group-size N puts up to N consecutive
segments of the same document into one request (0 = no limit, i.e. whole documents), and --group-chars Y
caps the text of a request at about Y characters (default 8000, 0 = no limit). This saves the repeated
instructions and the per-request overhead and gives the model the neighbouring segments as context.
The model labels every error with the number of its segment, so the output is still per segment.
Grouped requests produce longer answers: raise --max-new-tokens (and --max-model-len for vllm) with
the group size. If a request fails, all its segments are retried by --resume.

Resuming: next to the output there is OUTPUT.done with one line per finished segment (including those
without any error, which are not in the output) and, on its first line, the settings of the run.
Run the same command with --resume to skip the finished segments and append to the output. Segments
whose request failed or whose answer was unparsable are not marked as finished, so --resume retries
them. Resuming with different settings (model, --show, --annotate, grouping, ...) is refused.
Without --resume, the output and OUTPUT.done are overwritten.

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
        --show all --annotate canary_asr canary_cs --domain "This is an earnings call." \\
        --group-size 8 --max-new-tokens 3000
    # after an interruption: the same command plus --resume
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import harm_llm
import llm_backend
import prompts

try:
    from tqdm import tqdm
except ImportError:  # optional: without it (or without a terminal) progress is logged as lines
    tqdm = None

PROGRESS_EVERY = 10        # line log (no tqdm): after every N processed segments ...
PROGRESS_SECONDS = 30.0    # ... or after N seconds, whichever comes first


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="output of 30_add_and_align_sentences.py")
    ap.add_argument("--output", required=True, help="harmful errors, one segment per line "
                                                    "(written gradually, flushed after every record)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: skip the segments listed in OUTPUT.done and "
                         "append to the output (refused if the settings differ)")
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
                    help="unit of annotation (document level is not implemented yet)")
    ap.add_argument("--group-size", type=int, default=1,
                    help="segments of one document per LLM request (default 1 = one request per "
                         "segment; 0 = no limit, whole documents, bounded by --group-chars)")
    ap.add_argument("--group-chars", type=int, default=8000,
                    help="max characters of segment text per request when grouping (default 8000; "
                         "0 = no limit)")
    ap.add_argument("--context", type=int, default=0,
                    help="also show N neighbouring sentences of the same document (not annotated); "
                         "with grouping only before the first and after the last segment of a request")
    ap.add_argument("--domain", help="static description of the situation, added to every prompt")
    ap.add_argument("--domain-file", help="read --domain from a file")
    ap.add_argument("--limit", type=int, help="only the first N segments (for testing)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="store the raw LLM answer in the output (with grouping: the answer of the "
                         "whole request)")
    ap.add_argument("--keep-gold-meta", action="store_true", help="copy \"gold_meta\" to the output")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    llm_backend.add_arguments(ap)
    return ap.parse_args()


# ---------------------------------------------------------------- resume

def segment_key(rec):
    return f"{rec.get('document')}\t{rec.get('beg')}\t{rec.get('end')}"


def config_header(args):
    """First line of the .done file: everything that changes what the output means."""
    cfg = {"model": args.model, "backend": args.backend, "level": args.level,
           "show": args.show, "annotate": args.annotate, "asr_system": args.asr_system,
           "gold_system": args.gold_system, "context": args.context, "domain": args.domain,
           "group_size": args.group_size, "group_chars": args.group_chars,
           "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
           "keep_raw": args.keep_raw, "keep_gold_meta": args.keep_gold_meta}
    return "#config " + json.dumps(cfg, sort_keys=True, ensure_ascii=False)


def load_done(done_path, output_path, header):
    """Keys of the finished segments (empty set if there is nothing to resume from)."""
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if not lines or lines[0] != header:
            old = lines[0][len("#config "):] if lines else "(empty file)"
            sys.exit(f"--resume: the settings differ from the interrupted run, refusing to mix "
                     f"results.\n  before: {old}\n  now:    {header[len('#config '):]}\n"
                     f"Use the same settings, or run without --resume to start over.")
        return {l for l in lines[1:] if l.strip()}
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        sys.exit(f"--resume: {output_path} exists but {done_path} does not, so it is unknown which "
                 f"segments are finished. Run without --resume to start over.")
    return set()


# ---------------------------------------------------------------- progress

def fmt_time(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def n_errors(stats):
    return sum(v for k, v in stats.items() if k.startswith("error in"))


def log_progress(done, total, t0, n_out, stats):
    """One progress line (used when there is no tqdm bar)."""
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = fmt_time((total - done) / rate) if rate > 0 else "?"
    msg = (f"[{time.strftime('%H:%M:%S')}] {done}/{total} ({100 * done / total:.1f}%) "
           f"| {rate:.2f} seg/s | elapsed {fmt_time(elapsed)} | ETA {eta} "
           f"| flagged segments: {n_out}, errors: {n_errors(stats)}")
    if stats.get("unparsable answer"):
        msg += f", unparsable: {stats['unparsable answer']}"
    if stats.get("failed request"):
        msg += f", FAILED: {stats['failed request']}"
    print(msg, file=sys.stderr, flush=True)


def make_bar(total):
    """A tqdm bar if stderr is a terminal and tqdm is installed, else None (line log instead)."""
    if tqdm is None or not sys.stderr.isatty():
        return None
    return tqdm(total=total, unit="seg", file=sys.stderr, dynamic_ncols=True, smoothing=0.1)


def bar_postfix(bar, n_out, stats):
    postfix = {"flagged": n_out, "errors": n_errors(stats)}
    if stats.get("unparsable answer"):
        postfix["unparsable"] = stats["unparsable answer"]
    if stats.get("failed request"):
        postfix["FAILED"] = stats["failed request"]
    bar.set_postfix(postfix, refresh=False)


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    if args.level != "segment":
        sys.exit("--level document is not implemented yet")
    if args.group_size < 0 or args.group_chars < 0:
        sys.exit("--group-size and --group-chars must not be negative")
    if args.domain_file:
        args.domain = Path(args.domain_file).read_text(encoding="utf-8").strip()
    grouped = args.group_size != 1

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
          file=sys.stderr, flush=True)

    done_path = args.output + ".done"
    header = config_header(args)
    mode = "w"
    if args.resume and not args.dry_run:
        done_keys = load_done(done_path, args.output, header)
        if done_keys:
            mode = "a"
            before = len(work)
            work = [(ctx, v) for ctx, v in work if segment_key(ctx["record"]) not in done_keys]
            print(f"resuming: {before - len(work)} segments already done, {len(work)} left",
                  file=sys.stderr, flush=True)
            if not work:
                print("nothing left to do", file=sys.stderr)
                return

    groups = harm_llm.build_groups(work, args)
    if grouped:
        chats = [prompts.build_group_chat([v for _, v in g]) for g in groups]
    else:
        chats = [prompts.build_chat(g[0][1]) for g in groups]
    if args.dry_run:
        print(chats[0][0]["content"] + "\n\n" + chats[0][1]["content"])
        return

    total = len(work)
    biggest = max(len(g) for g in groups)
    print(f"{total} segments in {len(groups)} requests (up to {biggest} segments per request)",
          file=sys.stderr, flush=True)
    if grouped and args.max_new_tokens < 2048:
        print(f"note: --max-new-tokens {args.max_new_tokens} may truncate the answer for "
              f"{biggest} segments per request (unparsable answers); consider 2048 or more",
              file=sys.stderr, flush=True)

    backend = llm_backend.get_backend(llm_backend.BackendConfig.from_args(args))
    print(f"annotating with {args.model} ({args.backend}); writing to {args.output} as results arrive",
          file=sys.stderr, flush=True)

    stats = Counter()
    n_out = 0
    done = 0
    t0 = time.time()
    last_log, last_log_done = t0, 0
    interrupted = False
    bar = make_bar(total)
    # line-buffered + explicit flush: every finished record is on disk immediately
    try:
        with open(args.output, mode, encoding="utf-8", buffering=1) as out, \
                open(done_path, mode, encoding="utf-8", buffering=1) as done_file:
            if mode == "w":
                done_file.write(header + "\n")
                done_file.flush()
            try:
                answers = backend.generate_iter(chats, prompts.response_schema(systems, grouped))
                for group, answer in zip(groups, answers):
                    n = len(group)
                    done += n
                    data = None
                    if answer is None:
                        stats["failed request"] += n          # counted in segments
                    else:
                        data = harm_llm.extract_json(answer)
                        if data is None:
                            stats["unparsable answer"] += n   # counted in segments
                    if data is not None:
                        # distribute the annotations over the segments of the group
                        per_segment = defaultdict(list)
                        for a in data["annotations"]:
                            try:
                                k = int(a.get("segment", 1)) if grouped else 1
                            except (TypeError, ValueError):
                                k = 0
                            if not 1 <= k <= n:
                                stats["bad segment number"] += 1
                                continue
                            per_segment[k].append(a)
                        for k, (ctx, view) in enumerate(group, 1):
                            ctx["shown"] = view["shown"]
                            entries = [e for e in (harm_llm.clean_annotation(a, ctx, stats)
                                                   for a in per_segment[k]) if e]
                            for e in entries:
                                stats[f"error in {e['system']}"] += 1
                            rec = harm_llm.build_output(ctx, entries, args, raw=answer)
                            if rec:
                                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                out.flush()
                                n_out += 1
                            # finished: written after the record, so a crash in between duplicates
                            # a record rather than losing one
                            done_file.write(segment_key(ctx["record"]) + "\n")
                        done_file.flush()

                    if bar is not None:
                        bar_postfix(bar, n_out, stats)
                        bar.update(n)
                    else:
                        now = time.time()
                        if (done - last_log_done >= PROGRESS_EVERY or done == total
                                or now - last_log >= PROGRESS_SECONDS):
                            log_progress(done, total, t0, n_out, stats)
                            last_log, last_log_done = now, done
            except KeyboardInterrupt:
                interrupted = True
    finally:
        if bar is not None:
            bar.close()

    if interrupted:
        print(f"interrupted after {done}/{total} segments; the output so far is saved, "
              f"continue with --resume", file=sys.stderr, flush=True)
    print(f"{'(partial) ' if interrupted else ''}-> {args.output}: {n_out} segment(s) with harmful "
          f"errors ({n_errors(stats)} error(s)) from {done}/{total} segments in "
          f"{fmt_time(time.time() - t0)}", file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}", file=sys.stderr)
    retry = stats["failed request"] + stats["unparsable answer"]
    if retry:
        print(f"  {retry} segment(s) not marked as finished; run again with --resume to retry them",
              file=sys.stderr)


if __name__ == "__main__":
    main()