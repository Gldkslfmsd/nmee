#!/usr/bin/env python3
"""
Streaming LLM translation only (NeMo@IWSLT 2026 cascade, vLLM): reads the JSONL files written by
nemo_streaming_parakeet.py and writes SimulStreaming-style JSONL with the translation, for one or more
input files and one or more target languages in sequence.

    pip install vllm        # plus a NeMo version that has nemo.collections.asr.inference.nmt
    python nemo_streaming_llm.py -t de -o out/ out/*.en.jsonl
    python nemo_streaming_llm.py -t de cs it --nmt-model Qwen/Qwen3.5-27B -o mt/ asr/*.en.jsonl

One output per (input, target): <outdir>/<docid>.<tgt>.jsonl. docid is the input file name without
".jsonl" and without a trailing ".<src>" (doc.en.jsonl -> doc). Existing outputs are skipped (--overwrite
regenerates them); each output is guarded by a <output>.lock directory, so several jobs can share one
work list in parallel (--no-lock disables this). Files are written to <name>.part and renamed when done.

The input must come from nemo_streaming_parakeet.py: one line per ASR step, with the confirmed "text" and the
"unconfirmed_text" tail. Each line is replayed at its "emission_time" (simulated clock) and gives one LLM step
on the current ASR hypothesis (confirmed + unconfirmed text; --mt-input committed: confirmed text only), with
the already fixed translation as forced prefix and the last finished segment as context. The fixed prefix
(NeMo's longest-common-prefix rule) is the confirmed "text" of the output, the rest of the current translation
its "unconfirmed_text". At each is_final line the segment is translated completely. Output emission_time
equals the input emission_time (computation-unaware simulation, the only mode implemented). This reproduces the cascade
of nemo_streaming_cascade.py, which is the same code run in one process.

WARNING: computation-UNAWARE simulation only. emission_time is the audio time of the ASR step and ignores
the compute time of ASR and LLM (computation-aware simulation is not implemented).
"""
import argparse
import sys
from pathlib import Path

from utils import (
    AsrEventFeeder, Clock, add_mt_input_arg, JsonlWriter, StreamingTranslator, add_io_args, add_llm_args,
    atomic_out, build_llm, check_inputs, claim, log, read_jsonl, resolve_llm_args,
    silence_resource_tracker, warn_comp_unaware,
)


def docid_of(path, src_code):
    name = path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
    return name[: -len(src_code) - 1] if name.endswith("." + src_code) else name


def main():
    silence_resource_tracker()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "ASR JSONL file(s) from nemo_streaming_parakeet.py")
    add_llm_args(ap, default_gpu_mem=0.85)
    add_mt_input_arg(ap)
    args = ap.parse_args()
    warn_comp_unaware()
    check_inputs(args)
    resolve_llm_args(args)

    llm = None  # loaded when the first output needs it (all outputs may exist already)
    for inp in args.inputs:
        docid = docid_of(inp, args.src_code)
        for code, name in args.tgts:
            out = args.outdir / f"{docid}.{code}.jsonl"
            if out.resolve() == inp.resolve():
                log(f"skip: output would overwrite the input {inp}")
                continue
            with claim(out, args) as ok:
                if not ok:
                    continue
                if llm is None:
                    llm = build_llm(args)
                clock = Clock()
                last_ref = 0.0
                with atomic_out(out) as fh:
                    tr = StreamingTranslator(llm, args.src_name, name, JsonlWriter(fh), args.context_segments)
                    feeder = AsrEventFeeder(tr, use_unconfirmed=args.mt_input == "partial")
                    for ev in read_jsonl(inp):
                        last_ref = float(ev.get("emission_time", last_ref))
                        clock.wait_until(last_ref)
                        feeder.feed(ev, clock, last_ref)
                    tr.finish(feeder.last_start, feeder.last_end, clock, last_ref)
            log(f"{out}: {tr.llm_calls} LLM calls"
                + (f"; {tr.revisions} segment(s) where the final translation revised written words"
                   if tr.revisions else ""))


if __name__ == "__main__":
    sys.exit(main())