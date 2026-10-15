#!/usr/bin/env python3
"""
Streaming LLM translation only (NeMo@IWSLT 2026 cascade, vLLM): reads the JSONL written by
nemo_streaming_parakeet.py and writes SimulStreaming-style JSONL with the translation.

    pip install vllm        # plus a NeMo version that has nemo.collections.asr.inference.nmt
    python nemo_streaming_llm.py asr.jsonl --tgt-lang German -o mt.jsonl
    python nemo_streaming_parakeet.py a.wav --comp-unaware | python nemo_streaming_llm.py - --tgt-lang German

Each input line is replayed at its "emission_time" (simulated clock): the LLM re-translates the
growing source of the open segment, with the already fixed translation as forced prefix; the fixed
prefix (NeMo's longest-common-prefix rule) is written append-only. At each is_final line the segment
is translated completely. With --comp-unaware, output emission_time equals the input emission_time;
otherwise LLM compute time is added.
"""
import argparse
import sys

from utils import (
    AsrEventFeeder, Clock, JsonlWriter, StreamingTranslator, add_common_args, add_llm_args, build_llm,
    close_out, log, open_out, read_jsonl, resolve_llm_args, warmup_llm,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asr_jsonl", help="ASR JSONL from nemo_streaming_parakeet.py, '-' for stdin")
    ap.add_argument("-o", "--output", default="-", help="output JSONL, '-' for stdout (default)")
    add_llm_args(ap, default_gpu_mem=0.85)
    add_common_args(ap)
    args = ap.parse_args()
    if args.asr_jsonl == "-" and args.output == "-":
        pass  # fine: stdin -> stdout
    resolve_llm_args(args)

    llm = build_llm(args)
    if not args.no_warmup and not args.comp_unaware:
        warmup_llm(llm, args.src_lang, args.tgt_lang)

    clock = Clock(args.comp_unaware)
    out = open_out(args.output)
    tr = StreamingTranslator(llm, args.src_lang, args.tgt_lang, JsonlWriter(out), args.context_words)
    feeder = AsrEventFeeder(tr)
    last_ref = 0.0
    try:
        for ev in read_jsonl(args.asr_jsonl):
            last_ref = float(ev.get("emission_time", last_ref))
            clock.wait_until(last_ref)
            feeder.feed(ev, clock, last_ref)
        tr.finish(feeder.last_end, clock, last_ref)
    finally:
        close_out(out)
    log(f"{tr.llm_calls} LLM calls" + (f"; {tr.revisions} segment(s) where the final translation revised "
                                       "already-written words" if tr.revisions else ""))


if __name__ == "__main__":
    sys.exit(main())
