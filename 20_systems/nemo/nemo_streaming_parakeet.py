#!/usr/bin/env python3
"""
Streaming ASR only: nvidia/parakeet-unified-en-0.6b (NeMo pipeline API) -> SimulStreaming-style JSONL.

    pip install -U "nemo_toolkit[asr]" soundfile librosa
    python nemo_streaming_parakeet.py audio.wav -o asr.jsonl                 # --latency low (default)
    python nemo_streaming_parakeet.py audio.wav --latency high --comp-unaware > asr.jsonl

Output lines:
    {"start": 0.0, "end": 1.12, "text": " And so,", "words": [...], "is_final": false, "emission_time": 2.31}
    {"is_final": true, "emission_time": 5.02}        # end-of-utterance (EoU), no text

Text is append-only: a word is written once two consecutive partial hypotheses agree on it
(LocalAgreement-2); at EoU the rest of the segment is written and an is_final line follows.
"start"/"end" and per-word times are approximations (NeMo gives no timestamps for partial hypotheses).
Feed the output to nemo_streaming_llm.py for translation.
"""
import argparse
import sys

from utils import (
    Clock, Committer, JsonlWriter, add_asr_args, add_common_args, asr_request_options, asr_steps,
    audio_duration, build_asr_pipeline, close_out, log, open_out, resolve_asr_args, warmup_asr,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="audio file (anything soundfile/librosa reads; resampled to 16 kHz mono)")
    ap.add_argument("-o", "--output", default="-", help="output JSONL, '-' for stdout (default)")
    add_asr_args(ap)
    add_common_args(ap)
    args = ap.parse_args()
    resolve_asr_args(args)

    pipeline = build_asr_pipeline(args)
    opts = asr_request_options(args)
    if not args.no_warmup and not args.comp_unaware:
        warmup_asr(pipeline, opts)

    duration = audio_duration(args.audio)
    clock = Clock(args.comp_unaware)
    out = open_out(args.output)
    asr = Committer(JsonlWriter(out))
    try:
        for st in asr_steps(pipeline, opts, args.audio, args.chunk, duration, clock):
            asr.step(st.final_text, st.partial_text, st.audio_t, st.prev_audio_t, st.emission_time)
        asr.flush(duration, clock.end_emission(duration))
    finally:
        close_out(out)
    if asr.revisions:
        log(f"NOTE: {asr.revisions} segment(s) where the final transcript revised already-written words.")


if __name__ == "__main__":
    sys.exit(main())
