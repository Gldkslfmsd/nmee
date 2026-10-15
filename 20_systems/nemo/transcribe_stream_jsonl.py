#!/usr/bin/env python3
"""
Streaming ASR with nvidia/parakeet-unified-en-0.6b (NeMo pipeline API) that writes
SimulStreaming-style JSONL: append-only text updates, one JSON object per line.

Usage:
    pip install -U "nemo_toolkit[asr]" soundfile librosa
    python transcribe_stream_jsonl.py audio.wav -o out.jsonl
    python transcribe_stream_jsonl.py audio.wav --comp-unaware > out.jsonl

Output lines (same shape as SimulStreaming's default stdout):
    {"start": 0.0, "end": 1.12, "text": " And so,", "words": [{"start": .., "end": .., "text": " And"}, ...],
     "is_final": false, "emission_time": 2.31}
    {"is_final": true, "emission_time": 5.02}        # end-of-utterance (EoU), no text

How the append-only text is produced:
  NeMo's partial_transcript is unstable (it is revised as audio arrives), so a word is
  committed only when it is the same in two consecutive hypotheses (LocalAgreement-2,
  longest common prefix). At EoU, whatever is left of the segment is committed and an
  {"is_final": true} line follows.

Caveats:
  * NeMo gives no timestamps for partial hypotheses, so "start"/"end" and the per-word
    times are approximations: "end" is the audio time at which the words were committed
    (minus one chunk for LCP commits), and words are spread evenly between start and end.
  * There is no "tokens" field (it holds Whisper token ids in SimulStreaming).
"""
import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

CONFIG_URLS = [
    "https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml",
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml",
]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def get_config_path(path):
    """Use --config if given, else download NeMo's buffered_rnnt.yaml once and cache it."""
    if path:
        return path
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nemo_stream_jsonl")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, "buffered_rnnt.yaml")
    if os.path.exists(dst):
        return dst
    last_err = None
    for url in CONFIG_URLS:
        try:
            urllib.request.urlretrieve(url, dst)
            return dst
        except Exception as e:  # noqa: BLE001
            last_err = e
            if os.path.exists(dst):
                os.remove(dst)
    sys.exit(
        f"Could not download buffered_rnnt.yaml ({last_err}). Download "
        "examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml from the NeMo repo "
        "and pass it with --config."
    )


def build_pipeline(args, cfg_path):
    from omegaconf import OmegaConf
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    cfg = OmegaConf.load(cfg_path)
    overrides = {
        "asr.model_name": args.model,
        "asr.device": args.device,
        "asr.device_id": 0 if args.device == "cuda" else -1,
        "streaming.left_padding_size": args.left,
        "streaming.chunk_size": args.chunk,
        "streaming.right_padding_size": args.right,
        "streaming.batch_size": 1,
        "streaming.stateful": True,
        "asr_decoding_type": "rnnt",
        "asr.decoding.greedy.use_cuda_graph_decoder": False,
        "log_level": 40,
    }
    missing = object()
    for key, value in overrides.items():
        if OmegaConf.select(cfg, key, default=missing) is missing:
            sys.exit(f"Config {cfg_path} has no key '{key}' (NeMo version mismatch?).")
        OmegaConf.update(cfg, key, value)
    return PipelineBuilder.build_pipeline(cfg)


def cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def audio_duration(path):
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:  # noqa: BLE001
        import librosa

        return float(librosa.get_duration(path=path))


def warmup(pipeline, opts):
    """Run ~3 s of noise through the pipeline so CUDA init is not counted in latency."""
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "warmup.wav")
        sf.write(p, (np.random.randn(16000 * 3) * 0.01).astype("float32"), 16000)
        pipeline.run([p], options=[opts])


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class JsonlWriter:
    def __init__(self, fh):
        self.fh = fh

    def _write(self, obj):
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()

    def text(self, words, start, end, emission_time):
        n = len(words)
        step_words = []
        for i, w in enumerate(words):
            t = round(start + (end - start) * (i + 1) / n, 3)
            step_words.append({"start": t, "end": t, "text": " " + w})
        self._write(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": "".join(" " + w for w in words),
                "words": step_words,
                "is_final": False,
                "emission_time": round(emission_time, 4),
            }
        )

    def final(self, emission_time):
        self._write({"is_final": True, "emission_time": round(emission_time, 4)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="audio file (any format librosa/soundfile can read; resampled to 16 kHz mono)")
    ap.add_argument("-o", "--output", default="-", help="output JSONL file, '-' for stdout (default)")
    ap.add_argument("--model", default="nvidia/parakeet-unified-en-0.6b")
    ap.add_argument("--config", default=None, help="path to buffered_rnnt.yaml (downloaded if omitted)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--left", type=float, default=5.6, help="left context, s (model was trained with 5.6)")
    ap.add_argument("--chunk", type=float, default=0.56, help="chunk (frame) size, s; multiple of 0.08")
    ap.add_argument("--right", type=float, default=0.56, help="right context, s; multiple of 0.08")
    ap.add_argument("--eou-ms", type=int, default=1600, help="silence (ms) that closes an utterance (EoU)")
    ap.add_argument(
        "--comp-unaware",
        action="store_true",
        help="emission_time = audio time (no compute cost). Default: computation-aware simulated clock.",
    )
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    for name in ("chunk", "right"):
        v = getattr(args, name)
        if abs(v / 0.08 - round(v / 0.08)) > 1e-6:
            log(f"WARNING: --{name} {v}s is not a multiple of 80 ms; the model may not accept it.")
    log(f"latency (chunk + right) = {args.chunk + args.right:.2f}s")

    from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

    pipeline = build_pipeline(args, get_config_path(args.config))
    opts = ASRRequestOptions(stop_history_eou=args.eou_ms)
    if not args.no_warmup and not args.comp_unaware:
        warmup(pipeline, opts)

    duration = audio_duration(args.audio)
    out_fh = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    writer = JsonlWriter(out_fh)

    request_generator = pipeline.get_request_generator()
    request_generator.set_audio_filepaths([args.audio], [opts])

    committed = []      # words committed since the last EoU
    prev_hyp = []       # previous partial hypothesis (words)
    last_end = 0.0      # audio time of the last emitted text
    prev_audio_t = 0.0  # audio time at the previous step
    clock = 0.0         # simulated wall clock (computation-aware)
    revisions = 0       # committed text later contradicted by the EoU transcript

    pipeline.open_session()
    try:
        for step, requests in enumerate(request_generator):
            audio_t = min((step + 1) * args.chunk, duration)
            clock = max(clock, audio_t)  # cannot process audio before it has arrived

            cuda_sync()
            t0 = time.perf_counter()
            outputs = pipeline.transcribe_step(requests)
            cuda_sync()
            clock += time.perf_counter() - t0
            emission_time = audio_t if args.comp_unaware else clock

            for out in outputs:
                final = out.final_transcript.split() if out.final_transcript else []
                hyp = out.partial_transcript.split() if out.partial_transcript else []

                if final:  # end of utterance: flush the rest, close the segment
                    if final[: len(committed)] != committed:
                        revisions += 1
                    new = final[len(committed):]
                    if new:
                        start = last_end
                        end = max(audio_t, start)
                        writer.text(new, start, end, emission_time)
                        last_end = end
                    writer.final(emission_time)
                    committed, prev_hyp = [], []
                else:  # LocalAgreement-2: commit the prefix shared with the previous hypothesis
                    lcp = common_prefix_len(prev_hyp, hyp)
                    if lcp > len(committed):
                        new = hyp[len(committed):lcp]
                        start = last_end
                        end = max(prev_audio_t, start)
                        writer.text(new, start, end, emission_time)
                        last_end = end
                        committed = hyp[:lcp]
                    prev_hyp = hyp
            prev_audio_t = audio_t

        # end of audio: flush anything not yet committed
        if len(prev_hyp) > len(committed):
            end_t = duration
            emission_time = duration if args.comp_unaware else max(clock, duration)
            writer.text(prev_hyp[len(committed):], last_end, max(end_t, last_end), emission_time)
            writer.final(emission_time)
    finally:
        pipeline.close_session()
        if out_fh is not sys.stdout:
            out_fh.close()

    if revisions:
        log(f"NOTE: {revisions} utterance(s) where the final transcript revised already-committed words.")


if __name__ == "__main__":
    main()
