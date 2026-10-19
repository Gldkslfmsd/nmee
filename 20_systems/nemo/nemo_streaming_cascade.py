#!/usr/bin/env python3
"""
Full streaming cascade in one process: Parakeet ASR -> vLLM LLM translation (NeMo@IWSLT 2026), for one
or more audio files and one or more target languages in sequence. Writes two kinds of SimulStreaming-style
JSONL files per document: the ASR output and the translation.

    python nemo_streaming_cascade.py -t de --latency low  -o out/ audio/*.mp3
    python nemo_streaming_cascade.py -t de cs --latency high --nmt-model Qwen/Qwen3.5-27B --nmt-gpu-mem 0.8 \\
        -o out/ talk.wav

Outputs in <outdir>: <docid>.<src>.jsonl (ASR, e.g. talk.en.jsonl) and <docid>.<tgt>.jsonl (translation,
e.g. talk.de.jsonl). Existing outputs are skipped (--overwrite regenerates them); each translation is
guarded by a <output>.lock directory, so several jobs can share one work list in parallel (--no-lock
disables this). Files are written to <name>.part and renamed when finished. The ASR file is written once
per document (by whichever job gets there first); with several targets the ASR is simply re-run for each.

The LLM translates the current ASR hypothesis at every ASR step, as in NeMo and the paper; --mt-input
committed makes it see only the confirmed ASR text. The ASR file is exactly what nemo_streaming_parakeet.py
writes, and the translation is what nemo_streaming_llm.py makes from it: this is the same code in one process.

ASR and vLLM share the GPU (the vLLM engine runs in this process), hence --nmt-gpu-mem 0.5 by default;
for large LLMs raise it (Qwen3.5-27B: about 0.8).

WARNING: computation-UNAWARE simulation only. emission_time is the audio time of the ASR step and ignores
the compute time of ASR and LLM (computation-aware simulation is not implemented).
"""
import argparse
import contextlib
import os
import sys

from utils import (
    MAX_NEW_TOKENS,
    AsrEventFeeder, Clock, add_mt_input_arg, Committer, JsonlWriter, StreamingTranslator, add_asr_args,
    add_io_args, add_llm_args, asr_request_options, asr_steps, atomic_out, audio_duration,
    build_asr_pipeline, build_llm, check_inputs, claim, log, resolve_asr_args, resolve_llm_args,
    silence_resource_tracker, warn_comp_unaware,
)


def main():
    silence_resource_tracker()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "input audio file(s) (resampled to 16 kHz mono)")
    add_mt_input_arg(ap)
    add_asr_args(ap, with_src_lang=False)
    add_llm_args(ap, default_gpu_mem=0.5)
    args = ap.parse_args()
    warn_comp_unaware()
    check_inputs(args)
    resolve_asr_args(args)
    resolve_llm_args(args)
    partial_mode = args.mt_input == "partial"

    pipeline = opts = llm = None  # loaded when the first output needs them
    for inp in args.inputs:
        docid = inp.stem
        asr_out = args.outdir / f"{docid}.{args.src_code}.jsonl"
        asr_handled = False  # the ASR file is written once per document by this process
        for code, name in args.tgts:
            mt_out = args.outdir / f"{docid}.{code}.jsonl"
            with claim(mt_out, args) as ok:
                if not ok:
                    continue
                if pipeline is None:
                    # ASR first: building the LLM sets CUDA_VISIBLE_DEVICES (NeMo's LLMTranslator.load_model)
                    pipeline = build_asr_pipeline(args)
                    opts = asr_request_options(args)
                    llm = build_llm(args)
                duration = audio_duration(str(inp))
                clock = Clock()
                with contextlib.ExitStack() as stack:
                    mt_fh = stack.enter_context(atomic_out(mt_out))
                    if not asr_handled and (args.overwrite or not asr_out.exists()):
                        asr_fh = stack.enter_context(atomic_out(asr_out))
                    else:  # already there: the ASR still runs (it drives the translation), output discarded
                        asr_fh = stack.enter_context(open(os.devnull, "w"))
                    asr_handled = True
                    asr = Committer(JsonlWriter(asr_fh))
                    tr = StreamingTranslator(llm, args.src_name, name, JsonlWriter(mt_fh), args.context_segments,
                                             max_prompt_tokens=args.max_model_len - MAX_NEW_TOKENS - 8)
                    feeder = AsrEventFeeder(tr, use_unconfirmed=partial_mode)
                    for st in asr_steps(pipeline, opts, str(inp), args.chunk, duration, clock):
                        for ev in asr.step(st.final_text, st.partial_text, st.prev_audio_t, st.audio_t,
                                           st.emission_time):
                            feeder.feed(ev, clock, st.emission_time)
                    end_em = clock.end_emission(duration)
                    for ev in asr.flush(duration, end_em):
                        feeder.feed(ev, clock, end_em)
                    tr.finish(feeder.last_start, duration, clock, end_em)
            log(f"{mt_out}: {tr.llm_calls} LLM calls, {duration:.1f}s of audio"
                + (f"; context dropped in {tr.trimmed} call(s)" if tr.trimmed else "")
                + (f"; {tr.splits} over-long segment(s) split" if tr.splits else "")
                + (f"; ASR revised written words in {asr.revisions} segment(s)" if asr.revisions else "")
                + (f"; {tr.revisions} MT segment(s) revised at EoU" if tr.revisions else ""))


if __name__ == "__main__":
    sys.exit(main())