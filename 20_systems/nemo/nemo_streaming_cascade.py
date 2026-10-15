#!/usr/bin/env python3
"""
Full streaming cascade in one process: Parakeet ASR -> vLLM LLM translation (NeMo@IWSLT 2026), for one
or more audio files and one or more target languages in sequence. Writes two kinds of SimulStreaming-style
JSONL files per document: the ASR output and the translation.

    python nemo_streaming_cascade.py -t de --latency low  --comp-unaware -o out/ audio/*.mp3
    python nemo_streaming_cascade.py -t de cs --latency high --nmt-model Qwen/Qwen3.5-27B --nmt-gpu-mem 0.8 \\
        -o out/ talk.wav

Outputs in <outdir>: <docid>.<src>.jsonl (ASR, e.g. talk.en.jsonl) and <docid>.<tgt>.jsonl (translation,
e.g. talk.de.jsonl). Existing outputs are skipped (--overwrite regenerates them); each translation is
guarded by a <output>.lock directory, so several jobs can share one work list in parallel (--no-lock
disables this). Files are written to <name>.part and renamed when finished. The ASR file is written once
per document (by whichever job gets there first); with several targets the ASR is simply re-run for each.

--mt-input partial   (default, as in the paper) the LLM sees NeMo's current unstable ASR hypothesis;
                     the translation prefix is fixed by NeMo's longest-common-prefix rule.
--mt-input committed the LLM sees only ASR words already written to the ASR JSONL (the result is the
                     same as running nemo_streaming_parakeet.py and nemo_streaming_llm.py in sequence).

ASR and vLLM share the GPU (the vLLM engine runs in this process), hence --nmt-gpu-mem 0.5 by default;
for large LLMs raise it (Qwen3.5-27B: about 0.8).
"""
import argparse
import contextlib
import os
import sys

from utils import (
    AsrEventFeeder, Clock, Committer, JsonlWriter, StreamingTranslator, add_asr_args, add_common_args,
    add_io_args, add_llm_args, asr_request_options, asr_steps, atomic_out, audio_duration,
    build_asr_pipeline, build_llm, check_inputs, claim, log, resolve_asr_args, resolve_llm_args,
    silence_resource_tracker, warmup_asr, warmup_llm,
)


def main():
    silence_resource_tracker()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "input audio file(s) (resampled to 16 kHz mono)")
    ap.add_argument("--mt-input", choices=["partial", "committed"], default="partial",
                    help="what the LLM translates: NeMo's current partial hypothesis (default) or only the "
                    "ASR words already committed")
    add_asr_args(ap, with_src_lang=False)
    add_llm_args(ap, default_gpu_mem=0.5)
    add_common_args(ap)
    args = ap.parse_args()
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
                    if not args.no_warmup and not args.comp_unaware:
                        warmup_asr(pipeline, opts)
                        warmup_llm(llm, args.src_name, args.tgts[0][1])
                duration = audio_duration(str(inp))
                clock = Clock(args.comp_unaware)
                with contextlib.ExitStack() as stack:
                    mt_fh = stack.enter_context(atomic_out(mt_out))
                    if not asr_handled and (args.overwrite or not asr_out.exists()):
                        asr_fh = stack.enter_context(atomic_out(asr_out))
                    else:  # already there: the ASR still runs (it drives the translation), output discarded
                        asr_fh = stack.enter_context(open(os.devnull, "w"))
                    asr_handled = True
                    asr = Committer(JsonlWriter(asr_fh))
                    tr = StreamingTranslator(llm, args.src_name, name, JsonlWriter(mt_fh), args.context_words)
                    feeder = AsrEventFeeder(tr)
                    for st in asr_steps(pipeline, opts, str(inp), args.chunk, duration, clock):
                        events = asr.step(st.final_text, st.partial_text, st.audio_t, st.prev_audio_t,
                                          st.emission_time)
                        if partial_mode:
                            if st.final_text:
                                tr.finalize(st.final_text, st.audio_t, clock, st.emission_time)
                            else:
                                tr.update(st.partial_text, st.prev_audio_t, clock, st.emission_time)
                        else:
                            for ev in events:
                                feeder.feed(ev, clock, st.emission_time)
                    end_em = clock.end_emission(duration)
                    for ev in asr.flush(duration, end_em):
                        if not partial_mode:
                            feeder.feed(ev, clock, end_em)
                    tr.finish(duration, clock, end_em)
            log(f"{mt_out}: {tr.llm_calls} LLM calls, {duration:.1f}s of audio"
                + (f"; ASR revised written words in {asr.revisions} segment(s)" if asr.revisions else "")
                + (f"; {tr.revisions} MT segment(s) revised at EoU" if tr.revisions else ""))


if __name__ == "__main__":
    sys.exit(main())