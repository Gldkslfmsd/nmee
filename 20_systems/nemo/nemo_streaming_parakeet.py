#!/usr/bin/env python3
"""
Streaming ASR only: nvidia/parakeet-unified-en-0.6b (NeMo pipeline API) -> SimulStreaming-style JSONL,
for one or more audio files in sequence.

    pip install -U "nemo_toolkit[asr]" soundfile librosa
    python nemo_streaming_parakeet.py -o out/ audio/*.mp3                  # --latency low (default)
    python nemo_streaming_parakeet.py --latency high -o out/ talk.wav

One output per input file: <outdir>/<docid>.<src>.jsonl (docid = file name without extension, src =
--src-lang, default en), written to <name>.part first and renamed when the file is finished. Existing
outputs are skipped (--overwrite regenerates them); each output is guarded by a <output>.lock
directory, so several jobs can share one work list in parallel (--no-lock disables this).

Output: one line per ASR step (a step = one chunk of audio) that has a hypothesis:
    {"emission_time": 5.76, "start": 2.88, "end": 4.32, "text": " as Mark",
     "unconfirmed_text": " said, I mean", "is_final": false}
  text              words confirmed in this step (increment, append-only): a word is confirmed once two
                    consecutive partial hypotheses agree on it (LocalAgreement-2)
  unconfirmed_text  the rest of the current hypothesis of the open segment (may still change)
  start, end        the audio that arrived since the previous step (one chunk), in seconds; emission_time:
                    simulated time of the line
  is_final          end of utterance (EoU, a pause of --eou-ms): "text" holds the remainder of the segment
Steps without any change are written too, because the LLM runs one step per ASR step. There are no
per-word times (NeMo gives none for partial hypotheses): start/end are the chunk, not the span of the text.
Feed the output to nemo_streaming_llm.py for translation.
"""
import argparse
import sys

from utils import (
    Clock, Committer, JsonlWriter, add_asr_args, add_common_args, add_io_args, asr_request_options,
    asr_steps, atomic_out, audio_duration, build_asr_pipeline, check_inputs, claim, log, resolve_asr_args,
    resolve_lang, silence_resource_tracker, warn_comp_unaware,
)


def main():
    silence_resource_tracker()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "input audio file(s) (anything soundfile/librosa reads; resampled to 16 kHz mono)")
    add_asr_args(ap)
    add_common_args(ap)
    args = ap.parse_args()
    warn_comp_unaware()
    check_inputs(args)
    resolve_asr_args(args)
    src_code = resolve_lang(args.src_lang)[0]

    pipeline = opts = None  # built when the first file needs it (all outputs may exist already)
    for inp in args.inputs:
        out = args.outdir / f"{inp.stem}.{src_code}.jsonl"
        with claim(out, args) as ok:
            if not ok:
                continue
            if pipeline is None:
                pipeline = build_asr_pipeline(args)
                opts = asr_request_options(args)
            duration = audio_duration(str(inp))
            clock = Clock()
            n_lines = 0
            with atomic_out(out) as fh:
                asr = Committer(JsonlWriter(fh))
                for st in asr_steps(pipeline, opts, str(inp), args.chunk, duration, clock):
                    n_lines += len(asr.step(st.final_text, st.partial_text, st.prev_audio_t, st.audio_t,
                                            st.emission_time))
                n_lines += len(asr.flush(duration, clock.end_emission(duration)))
        log(f"{out}: {n_lines} lines from {duration:.1f}s of audio"
            + (f"; {asr.revisions} segment(s) where the final transcript revised written words"
               if asr.revisions else ""))


if __name__ == "__main__":
    sys.exit(main())