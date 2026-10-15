#!/usr/bin/env python3
"""
Full streaming cascade in one process: Parakeet ASR -> vLLM LLM translation (NeMo@IWSLT 2026).
Writes two SimulStreaming-style JSONL files: one with the ASR output, one with the translation.

    python nemo_streaming_cascade.py a.wav --tgt-lang German --latency low  --comp-unaware
    python nemo_streaming_cascade.py a.wav --tgt-lang German --latency high --asr-output a.asr.jsonl --mt-output a.mt.jsonl

--mt-input partial   (default, as in the paper) the LLM sees NeMo's current unstable ASR hypothesis;
                     the translation prefix is fixed by NeMo's longest-common-prefix rule.
--mt-input committed the LLM sees only ASR words already written to the ASR JSONL (the result is the
                     same as running nemo_streaming_parakeet.py and nemo_streaming_llm.py in sequence).

ASR and vLLM share GPU 0 by default, hence --nmt-gpu-mem 0.5; with several GPUs use --nmt-device-id 1
(and e.g. --nmt-gpu-mem 0.85).
"""
import argparse
import os
import sys

from utils import (
    AsrEventFeeder, Clock, Committer, JsonlWriter, StreamingTranslator, add_asr_args, add_common_args,
    add_llm_args, asr_request_options, asr_steps, audio_duration, build_asr_pipeline, build_llm,
    close_out, log, open_out, resolve_asr_args, resolve_llm_args, warmup_asr, warmup_llm,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="audio file (resampled to 16 kHz mono)")
    ap.add_argument("--asr-output", default=None, help="ASR JSONL (default: <audio stem>.asr.jsonl)")
    ap.add_argument("--mt-output", default=None, help="translation JSONL (default: <audio stem>.mt.jsonl)")
    ap.add_argument("--mt-input", choices=["partial", "committed"], default="partial",
                    help="what the LLM translates: NeMo's current partial hypothesis (default) or only the "
                    "ASR words already committed")
    add_asr_args(ap)
    add_llm_args(ap, default_gpu_mem=0.5, src_lang_arg=False)
    add_common_args(ap)
    args = ap.parse_args()
    stem = os.path.splitext(args.audio)[0]
    args.asr_output = args.asr_output or stem + ".asr.jsonl"
    args.mt_output = args.mt_output or stem + ".mt.jsonl"
    if args.asr_output == "-" and args.mt_output == "-":
        sys.exit("--asr-output and --mt-output cannot both be '-'.")
    resolve_asr_args(args)
    resolve_llm_args(args)

    # ASR first: building the LLM sets CUDA_VISIBLE_DEVICES (NeMo's LLMTranslator.load_model)
    pipeline = build_asr_pipeline(args)
    opts = asr_request_options(args)
    llm = build_llm(args)
    if not args.no_warmup and not args.comp_unaware:
        warmup_asr(pipeline, opts)
        warmup_llm(llm, args.src_lang, args.tgt_lang)

    duration = audio_duration(args.audio)
    clock = Clock(args.comp_unaware)
    asr_out, mt_out = open_out(args.asr_output), open_out(args.mt_output)
    asr = Committer(JsonlWriter(asr_out))
    tr = StreamingTranslator(llm, args.src_lang, args.tgt_lang, JsonlWriter(mt_out), args.context_words)
    feeder = AsrEventFeeder(tr)
    partial_mode = args.mt_input == "partial"
    try:
        for st in asr_steps(pipeline, opts, args.audio, args.chunk, duration, clock):
            events = asr.step(st.final_text, st.partial_text, st.audio_t, st.prev_audio_t, st.emission_time)
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
    finally:
        close_out(asr_out)
        close_out(mt_out)
    if asr.revisions:
        log(f"NOTE: {asr.revisions} ASR segment(s) where the final transcript revised written words.")
    log(f"{tr.llm_calls} LLM calls; {tr.revisions} MT segment(s) revised at EoU")
    log(f"wrote {args.asr_output} and {args.mt_output}")


if __name__ == "__main__":
    sys.exit(main())
