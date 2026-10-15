"""
Shared code for nemo_streaming_parakeet.py, nemo_streaming_llm.py and nemo_streaming_cascade.py.

Sections:
  * generic       : logging, simulated clock, JSONL reading/writing, append-only committer
  * ASR           : NeMo streaming pipeline (nvidia/parakeet-unified-en-0.6b) with latency presets
  * LLM MT        : NeMo's vLLM LLMTranslator driven incrementally (NeMo@IWSLT 2026 cascade)

JSONL format (same shape as SimulStreaming's default stdout), used for ASR and MT output:
    {"start": 0.0, "end": 1.12, "text": " And so,", "words": [{"start": .., "end": .., "text": " And"}, ...],
     "is_final": false, "emission_time": 2.31}
    {"is_final": true, "emission_time": 5.02}        # end of utterance (EoU), no text
"""
import contextlib
import json
import os
import socket
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CONFIG_URLS = [
    "https://raw.githubusercontent.com/NVIDIA-NeMo/NeMo/main/examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml",
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/main/examples/asr/conf/asr_streaming_inference/buffered_rnnt.yaml",
]

# ASR settings of NeMo's IWSLT 2026 En->De system (Grigoryan et al., 2026), in seconds / ms.
# In the paper's cascade they gave CU LongYAAL of about 1.9 s (low) and 3.2 s (high).
LATENCY_PRESETS = {
    "low": {"left": 5.6, "chunk": 1.44, "right": 1.44, "eou_ms": 1600},
    "high": {"left": 5.6, "chunk": 2.88, "right": 2.88, "eou_ms": 1600},
}

DEFAULT_NMT_MODEL = "utter-project/EuroLLM-9B-Instruct"


# --------------------------------------------------------------------------- generic


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


class Clock:
    """Simulated wall clock.

    Computation-aware (default): `now` advances by the measured compute time of every step and can
    never run ahead of the data it processes (wait_until). Computation-unaware: emission times are
    just the reference times given by the caller (audio time, or the upstream emission time).
    """

    def __init__(self, comp_unaware=False):
        self.comp_unaware = comp_unaware
        self.now = 0.0

    def wait_until(self, t):
        self.now = max(self.now, t)

    @contextlib.contextmanager
    def compute(self):
        cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            cuda_sync()
            self.now += time.perf_counter() - t0

    def emission(self, ref):
        return ref if self.comp_unaware else self.now

    def end_emission(self, duration):
        return duration if self.comp_unaware else max(self.now, duration)


def read_jsonl(path):
    """Yield the JSON objects of a JSONL file."""
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{n}: not valid JSON ({e})")


def silence_resource_tracker():
    """Hide the 'Exception ignored in ResourceTracker.__del__ ... _recursion_count' traceback that
    NeMo's `multiprocess` package prints at interpreter exit on Python 3.12 (harmless)."""
    for module in ("multiprocess.resource_tracker", "multiprocessing.resource_tracker"):
        try:
            tracker = __import__(module, fromlist=["ResourceTracker"]).ResourceTracker
        except (ImportError, AttributeError):
            continue
        original = tracker._stop

        def quiet_stop(self, *args, __original=original, **kwargs):
            try:
                return __original(self, *args, **kwargs)
            except Exception:  # noqa: BLE001
                return None

        tracker._stop = quiet_stop


# --------------------------------------------------------------------------- files, locks, languages

LANG_NAMES = {
    "en": "English", "de": "German", "cs": "Czech", "sk": "Slovak", "pl": "Polish", "it": "Italian",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "uk": "Ukrainian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "tr": "Turkish", "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian",
    "sl": "Slovenian", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "el": "Greek",
    "et": "Estonian", "lv": "Latvian", "lt": "Lithuanian", "hi": "Hindi", "no": "Norwegian",
}


def resolve_lang(x):
    """'de' or 'German' -> ('de', 'German'). The code names the output files, the name goes into the prompt."""
    x = x.strip()
    if x.lower() in LANG_NAMES:
        return x.lower(), LANG_NAMES[x.lower()]
    for code, name in LANG_NAMES.items():
        if name.lower() == x.lower():
            return code, name
    return x.lower(), x  # unknown: used as given


@contextlib.contextmanager
def file_lock(path, enabled=True):
    """mkdir-based lock next to `path` (<path>.lock), so several jobs can share a work list.
    Yields True if this process took the lock, False if another holds it (mkdir is atomic)."""
    if not enabled:
        yield True
        return
    lock = path.with_name(path.name + ".lock")
    try:
        os.mkdir(lock)
    except FileExistsError:
        yield False
        return
    owner = lock / "owner"
    try:
        owner.write_text(f"{socket.gethostname()} pid {os.getpid()}\n")
        yield True
    finally:
        for fn in (owner.unlink, lambda: os.rmdir(lock)):
            try:
                fn()
            except OSError:
                pass


@contextlib.contextmanager
def claim(out, args):
    """Yields True if this process should produce `out`: it does not exist yet (or --overwrite) and
    no other job holds its lock."""
    if out.exists() and not args.overwrite:
        log(f"skip, exists: {out}")
        yield False
        return
    with file_lock(out, enabled=not args.no_lock) as acquired:
        if not acquired:
            log(f"skip, locked by another job: {out}")
            yield False
        elif out.exists() and not args.overwrite:  # finished by another job meanwhile
            log(f"skip, exists: {out}")
            yield False
        else:
            yield True


@contextlib.contextmanager
def atomic_out(path):
    """Text file handle for `path`: written to <path>.part (readable with tail -f while running) and
    renamed to `path` only when the block ends normally, so an interrupted run never leaves a partial
    file that a later skip-if-exists check would take for a finished one."""
    tmp = path.with_name(path.name + ".part")
    fh = tmp.open("w", encoding="utf-8")
    try:
        yield fh
        fh.flush()
        os.fsync(fh.fileno())
    except BaseException:
        fh.close()
        raise
    fh.close()
    os.replace(tmp, path)


def add_io_args(ap, inputs_help):
    ap.add_argument("inputs", nargs="+", type=Path, help=inputs_help)
    ap.add_argument("-o", "--outdir", "--output-dir", dest="outdir", type=Path, default=Path("."),
                    help="directory for the output .jsonl files (default: current directory)")
    ap.add_argument("--overwrite", action="store_true", help="regenerate outputs that already exist")
    ap.add_argument("--no-lock", action="store_true",
                    help="do not take a per-output lock directory (<output>.lock); use only when a single "
                    "job runs over these outputs. Without it, several jobs can share one work list.")


def check_inputs(args):
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        sys.exit("no such file(s): " + ", ".join(str(p) for p in missing))
    args.outdir.mkdir(parents=True, exist_ok=True)


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class JsonlWriter:
    """Writes SimulStreaming-style lines; every method also returns the dict it wrote."""

    def __init__(self, fh):
        self.fh = fh

    def _write(self, obj):
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()
        return obj

    def text(self, words, start, end, emission_time):
        n = len(words)
        step_words = []
        for i, w in enumerate(words):
            t = round(start + (end - start) * (i + 1) / n, 3)
            step_words.append({"start": t, "end": t, "text": " " + w})
        return self._write(
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
        return self._write({"is_final": True, "emission_time": round(emission_time, 4)})


class Committer:
    """Turns unstable (partial, final) ASR hypotheses into append-only JSONL updates.

    LocalAgreement-2: a word is committed once two consecutive partial hypotheses agree on it.
    At EoU (non-empty final text) the rest of the final text is committed and the segment closed.
    NeMo gives no timestamps for partial hypotheses, so "start"/"end" are approximations: "end" is
    the audio time at which the words were committed (minus one chunk for agreement commits).
    """

    def __init__(self, writer):
        self.writer = writer
        self.committed = []  # words committed since the last EoU
        self.prev = []       # previous partial hypothesis (words)
        self.last_end = 0.0  # audio time of the last emitted text
        self.revisions = 0   # committed text later contradicted by the EoU text

    def _emit(self, words, end, emission_time):
        start = self.last_end
        end = max(end, start)
        ev = self.writer.text(words, start, end, emission_time)
        self.last_end = end
        return ev

    def step(self, final_text, partial_text, audio_t, prev_audio_t, emission_time):
        """Returns the list of JSONL events (dicts) written by this step."""
        events = []
        final = final_text.split() if final_text else []
        hyp = partial_text.split() if partial_text else []

        if final:  # end of utterance: flush the rest, close the segment
            if final[: len(self.committed)] != self.committed:
                self.revisions += 1
            new = final[len(self.committed):]
            if new:
                events.append(self._emit(new, audio_t, emission_time))
            events.append(self.writer.final(emission_time))
            self.committed, self.prev = [], []
        else:
            lcp = common_prefix_len(self.prev, hyp)
            if lcp > len(self.committed):
                events.append(self._emit(hyp[len(self.committed):lcp], prev_audio_t, emission_time))
                self.committed = hyp[:lcp]
            self.prev = hyp
        return events

    def flush(self, duration, emission_time):
        """End of audio: commit whatever is pending and close an open segment."""
        events = []
        if self.prev or self.committed:
            if len(self.prev) > len(self.committed):
                events.append(self._emit(self.prev[len(self.committed):], duration, emission_time))
            events.append(self.writer.final(emission_time))
            self.committed, self.prev = [], []
        return events


# --------------------------------------------------------------------------- ASR


def add_common_args(ap):
    ap.add_argument(
        "--comp-unaware",
        action="store_true",
        help="computation-unaware simulation: emission_time ignores compute time. "
        "Default: computation-aware simulated clock.",
    )
    ap.add_argument("--no-warmup", action="store_true", help="skip the warm-up pass (computation-aware mode)")


def add_asr_args(ap, with_src_lang=True):
    g = ap.add_argument_group("ASR (NeMo streaming pipeline)")
    g.add_argument("--model", default="nvidia/parakeet-unified-en-0.6b")
    if with_src_lang:
        g.add_argument("-s", "--src-lang", default="en", metavar="CODE",
                       help="source language code, used only in the output file name <docid>.<code>.jsonl "
                       "(default: en; the model is English-only)")
    g.add_argument("--config", default=None, help="path to buffered_rnnt.yaml (downloaded and cached if omitted)")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    g.add_argument(
        "--latency",
        choices=sorted(LATENCY_PRESETS),
        default="low",
        help="latency preset (default: low). low: left 5.6 s, chunk 1.44 s, right 1.44 s, EoU 1600 ms "
        "(IWSLT low latency, under 2 s). high: chunk 2.88 s, right 2.88 s (IWSLT high latency, up to 4 s). "
        "Explicit --left/--chunk/--right/--eou-ms override the preset.",
    )
    g.add_argument("--left", type=float, default=None, help="left context, s (model was trained with 5.6)")
    g.add_argument("--chunk", type=float, default=None, help="chunk (frame) size, s; multiple of 0.08")
    g.add_argument("--right", type=float, default=None, help="right context, s; multiple of 0.08")
    g.add_argument("--eou-ms", type=int, default=None, help="silence (ms) that closes an utterance (EoU)")


def resolve_asr_args(args):
    for key, value in LATENCY_PRESETS[args.latency].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    log(
        f"preset '{args.latency}': left={args.left}s chunk={args.chunk}s "
        f"right={args.right}s eou={args.eou_ms}ms"
    )
    for name in ("chunk", "right"):
        v = getattr(args, name)
        if abs(v / 0.08 - round(v / 0.08)) > 1e-6:
            log(f"WARNING: --{name} {v}s is not a multiple of 80 ms; the model may not accept it.")
    log(f"latency (chunk + right) = {args.chunk + args.right:.2f}s")


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


def build_asr_pipeline(args):
    """ASR-only NeMo streaming pipeline (NeMo's built-in translation stays disabled)."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    cfg_path = get_config_path(args.config)
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
    if OmegaConf.select(cfg, "enable_nmt", default=missing) is not missing:
        OmegaConf.update(cfg, "enable_nmt", False)
    return PipelineBuilder.build_pipeline(cfg)


def asr_request_options(args):
    from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

    return ASRRequestOptions(stop_history_eou=args.eou_ms)


def audio_duration(path):
    try:
        import soundfile as sf

        return float(sf.info(path).duration)
    except Exception:  # noqa: BLE001
        import librosa

        return float(librosa.get_duration(path=path))


def warmup_asr(pipeline, opts):
    """Run ~3 s of noise through the pipeline so CUDA init is not counted in latency."""
    import numpy as np
    import soundfile as sf

    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "warmup.wav")
            sf.write(p, (np.random.randn(16000 * 3) * 0.01).astype("float32"), 16000)
            pipeline.run([p], options=[opts])
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: ASR warm-up failed ({e!r}); continuing without it.")


@dataclass
class AsrStep:
    audio_t: float         # audio seconds received so far
    prev_audio_t: float    # audio seconds at the previous step
    emission_time: float   # simulated time at which this ASR output exists
    final_text: str        # non-empty only at EoU: the finalized text of the segment
    partial_text: str      # current (unstable) hypothesis of the open segment


def asr_steps(pipeline, opts, audio, chunk, duration, clock):
    """Stream `audio` through the pipeline chunk by chunk, yielding one AsrStep per output.

    `clock` is advanced by the ASR compute time; a consumer that does more work per step (e.g. an
    LLM call) can add its own time with `with clock.compute():`.
    """
    request_generator = pipeline.get_request_generator()
    request_generator.set_audio_filepaths([audio], [opts])
    prev_audio_t = 0.0
    pipeline.open_session()
    try:
        for step, requests in enumerate(request_generator):
            audio_t = min((step + 1) * chunk, duration)
            clock.wait_until(audio_t)  # cannot process audio before it has arrived
            with clock.compute():
                outputs = pipeline.transcribe_step(requests)
            for out in outputs:
                yield AsrStep(
                    audio_t,
                    prev_audio_t,
                    clock.emission(audio_t),
                    out.final_transcript or "",
                    out.partial_transcript or "",
                )
            prev_audio_t = audio_t
    finally:
        pipeline.close_session()


# --------------------------------------------------------------------------- LLM MT


def add_llm_args(ap, default_gpu_mem):
    g = ap.add_argument_group("LLM translation (vLLM, via NeMo's LLMTranslator)")
    g.add_argument("-t", "--tgt-lang", nargs="+", required=True, metavar="LANG",
                   help="target language code(s), e.g. de cs it zh (or names such as German); one output file "
                   "<docid>.<code>.jsonl per language")
    g.add_argument("-s", "--src-lang", default="en", metavar="LANG",
                   help="source language code or name (default: en)")
    g.add_argument("--nmt-model", default=DEFAULT_NMT_MODEL,
                   help="HF id of the LLM (default: %(default)s, the best <=9B model for En->De in the paper). "
                   "NeMo itself accepts only EuroLLM-1.7B/9B-Instruct (9B may be gated on Hugging Face); Qwen models, "
                   "e.g. Qwen/Qwen3.5-27B (the IWSLT 2026 final En->De system), work through qwen_translator.py. "
                   "27B needs about 55 GB for weights: use --nmt-gpu-mem 0.8 or so.")
    g.add_argument("--nmt-device-id", type=int, default=0,
                   help="GPU index for the LLM (default: 0), counted among the GPUs in CUDA_VISIBLE_DEVICES. "
                   "NeMo sets CUDA_VISIBLE_DEVICES to it when loading vLLM. Caveat: the engine now runs in "
                   "this process, so if CUDA was already initialised (ASR in the cascade), the LLM uses the "
                   "same GPU as the ASR whatever this is; to separate them, use two scripts.")
    g.add_argument("--nmt-gpu-mem", type=float, default=default_gpu_mem,
                   help="vLLM gpu_memory_utilization, fraction of the whole GPU (default: %(default)s). "
                   "Negative = vLLM's own default.")
    g.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len (default: %(default)s)")
    g.add_argument("--llm-param", action="append", metavar="KEY=VALUE",
                   help="extra argument of vLLM's LLM(...), repeatable; VALUE is parsed as JSON if possible, "
                   "e.g. --llm-param 'limit_mm_per_prompt={\"image\":0,\"video\":0}' (skips the vision part of "
                   "Qwen3.5) or --llm-param enforce_eager=true")
    g.add_argument("--vllm-subprocess", action="store_true",
                   help="run vLLM's engine in a subprocess (vLLM's default; fails to init CUDA in some "
                   "environments with NeMo installed). Default here: in-process engine.")
    g.add_argument("--vllm-plugins", action="store_true",
                   help="load third-party vLLM plugins (NeMo registers one); off by default")
    g.add_argument("--max-num-seqs", type=int, default=8,
                   help="vLLM max_num_seqs (default: %(default)s). Only one request runs at a time here; vLLM's "
                   "default of 1024 makes Qwen3.5 fail with 'max_num_seqs exceeds available Mamba cache blocks' "
                   "and slows the CUDA-graph capture.")
    g.add_argument("--waitk", type=int, default=-1,
                   help="max lag in words between transcript and translation; -1 = longest common prefix "
                   "only (default, as in the paper)")
    g.add_argument("--context-words", type=int, default=100,
                   help="finished segments are given to the LLM as context (source and target side) until "
                   "about this many source words are reached (default: %(default)s; my choice, not from the paper)")


def resolve_llm_args(args):
    """args.src_code/src_name and args.tgts = [(code, name), ...] from -s/-t."""
    args.src_code, args.src_name = resolve_lang(args.src_lang)
    args.tgts = list(dict.fromkeys(resolve_lang(t) for t in args.tgt_lang))
    if args.nmt_gpu_mem is not None and args.nmt_gpu_mem < 0:
        args.nmt_gpu_mem = None
    log(f"translation: {args.src_name} -> {', '.join(n for _, n in args.tgts)} with {args.nmt_model} "
        f"(waitk={args.waitk})")


def build_llm(args):
    """NeMo's vLLM-based LLMTranslator (nemo.collections.asr.inference.nmt.llm_translator)."""
    # vLLM's engine subprocess can fail with "CUDA driver initialization failed" when NeMo's
    # `multiprocess` package is installed, so the engine runs in-process by default. NeMo's own vLLM
    # plugin is not needed here and can fail to import. Set the variables yourself to override.
    if not getattr(args, "vllm_subprocess", False):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if not getattr(args, "vllm_plugins", False):
        os.environ.setdefault("VLLM_PLUGINS", "")
    try:
        from nemo.collections.asr.inference.nmt.llm_translator import SUPPORTED_TRANSLATION_MODELS, LLMTranslator
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] == "nemo":
            sys.exit(f"Cannot import NeMo's LLM translator ({e}). It needs a NeMo version with the streaming "
                     "translation pipeline.")
        sys.exit(f"Cannot import NeMo's LLM translator, is vLLM installed? (`pip install vllm`): {e}")
    except ImportError as e:  # llm_translator raises ImportError("Failed to import vLLM.")
        sys.exit(f"Cannot import NeMo's LLM translator, is vLLM installed? (`pip install vllm`): {e}")

    translator_cls = LLMTranslator
    if args.nmt_model.split("/")[-1].lower().startswith("qwen"):
        # NeMo itself accepts only EuroLLM; qwen_translator.py adds Qwen (paper's prompt, chat template)
        from qwen_translator import QwenLLMTranslator as translator_cls
    elif args.nmt_model not in SUPPORTED_TRANSLATION_MODELS:
        sys.exit(
            f"--nmt-model {args.nmt_model} is not supported by the installed NeMo. "
            f"Supported: {', '.join(SUPPORTED_TRANSLATION_MODELS)}, and Qwen models (qwen_translator.py)"
        )
    # NeMo's load_model overwrites CUDA_VISIBLE_DEVICES with device_id, which would drop a restriction
    # set by the user or a scheduler. Treat --nmt-device-id as an index among the visible GPUs and
    # translate it to the physical id before NeMo overwrites the variable.
    device_id = args.nmt_device_id
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if not 0 <= device_id < len(ids):
            sys.exit(f"--nmt-device-id {device_id} but CUDA_VISIBLE_DEVICES={visible} exposes only "
                     f"{len(ids)} GPU(s) (valid: 0..{len(ids) - 1}).")
        if not ids[device_id].isdigit():
            sys.exit(f"CUDA_VISIBLE_DEVICES={visible} contains non-numeric entries; NeMo's LLMTranslator "
                     "needs a numeric id. Set CUDA_VISIBLE_DEVICES to plain GPU indices.")
        device_id = int(ids[device_id])
        log(f"LLM on physical GPU {device_id} (index {args.nmt_device_id} of CUDA_VISIBLE_DEVICES={visible})")
    # same defaults as NeMo's buffered_rnnt.yaml (nmt.llm_params / nmt.sampling_params)
    llm_params = {"dtype": "auto", "seed": 42, "max_model_len": args.max_model_len,
                  "max_num_seqs": args.max_num_seqs}
    if args.nmt_gpu_mem is not None:
        llm_params["gpu_memory_utilization"] = args.nmt_gpu_mem
    for item in args.llm_param or []:  # extra vLLM LLM(...) arguments, KEY=VALUE (VALUE is JSON if it parses)
        key, sep, value = item.partition("=")
        if not sep:
            sys.exit(f"--llm-param {item!r}: expected KEY=VALUE")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        llm_params[key] = value
    sampling_params = {"max_tokens": 100, "temperature": 0.0, "top_p": 0.9, "seed": 42}
    return translator_cls(
        model_name=args.nmt_model,
        source_language=args.src_name,
        target_language=args.tgts[0][1],
        waitk=args.waitk,
        device="cuda",
        device_id=device_id,
        batch_size=1,
        llm_params=llm_params,
        sampling_params=sampling_params,
    )


def warmup_llm(llm, src_lang, tgt_lang):
    try:
        llm.translate(["Hello, how are you?"], [""], [src_lang], [tgt_lang], [""], [""])
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: LLM warm-up failed ({e!r}); continuing without it.")


class StreamingTranslator:
    """Incremental LLM translation of a streaming transcript, following NeMo@IWSLT 2026.

    At every source update the LLM translates the current transcript of the open segment, with the
    already fixed translation prefix forced (pre-filled in the assistant turn) and the finished
    segments as context. The next prefix is NeMo's own rule (LLMTranslator.get_prefixes): the longest
    common prefix of two consecutive translations, cut back to a word boundary (and the waitk rule if
    --waitk > 0). Only that fixed prefix is written, so the output is append-only. At EoU the whole
    segment is translated, the rest is written, and prefix and contexts are updated.

    Methods take `clock` (a Clock) so that LLM compute time is simulated, `audio_t` (source audio
    time, written as "end") and `ref` (reference time for emission_time in computation-unaware mode).
    """

    def __init__(self, llm, src_lang, tgt_lang, writer, context_words=100):
        self.llm = llm
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.writer = writer
        self.context_words = context_words
        self.segments = []           # finished (source, translation) pairs, newest last
        self.prefix = ""             # fixed translation prefix of the open segment
        self.prev_translation = ""
        self.committed = []          # words already written for the open segment
        self.last_src = ""           # source text of the last LLM call (open segment)
        self.last_translation = ""
        self.last_end = 0.0
        self.revisions = 0           # written words later contradicted by the final translation
        self.llm_calls = 0

    def _contexts(self):
        src, tgt, n = [], [], 0
        for s, t in reversed(self.segments):
            src.append(s)
            tgt.append(t)
            n += len(s.split())
            if n >= self.context_words:
                break
        return " ".join(reversed(src)), " ".join(reversed(tgt))

    def _translate(self, src_text):
        if src_text == self.last_src and self.last_translation:
            return self.last_translation
        src_ctx, tgt_ctx = self._contexts()
        self.llm_calls += 1
        return self.llm.translate(
            [src_text], [self.prefix], [self.src_lang], [self.tgt_lang], [src_ctx], [tgt_ctx]
        )[0]

    def _emit(self, words, audio_t, emission_time):
        start = self.last_end
        end = max(audio_t, start)
        self.writer.text(words, start, end, emission_time)
        self.last_end = end

    def update(self, src_text, audio_t, clock, ref):
        """New (possibly unstable) source text of the open segment."""
        src_text = " ".join(src_text.split())
        if not src_text or src_text == self.last_src:
            return
        with clock.compute():
            translation = self._translate(src_text)
            new_prefix = self.llm.get_prefixes([src_text], [translation], [self.prev_translation])[0]
        self.last_src, self.last_translation = src_text, translation
        self.prev_translation = translation
        words = new_prefix.split()
        if len(words) > len(self.committed) and words[: len(self.committed)] == self.committed:
            self._emit(words[len(self.committed):], audio_t, clock.emission(ref))
            self.committed = words
        self.prefix = new_prefix

    def finalize(self, src_text, audio_t, clock, ref):
        """The open segment is complete (EoU): translate it fully, write the rest, reset."""
        src_text = " ".join(src_text.split())
        if not src_text:
            return
        with clock.compute():
            translation = self._translate(src_text)
        words = translation.split()
        if words[: len(self.committed)] != self.committed:
            self.revisions += 1
        new = words[len(self.committed):]
        if new:
            self._emit(new, audio_t, clock.emission(ref))
        self.writer.final(clock.emission(ref))
        self.segments.append((src_text, translation.strip()))
        self.segments = self.segments[-50:]
        self.prefix = self.prev_translation = self.last_src = self.last_translation = ""
        self.committed = []

    def finish(self, audio_t, clock, ref):
        """End of input: close a segment that never got an EoU."""
        if self.last_src:
            self.finalize(self.last_src, audio_t, clock, ref)


class AsrEventFeeder:
    """Feeds ASR JSONL events (as written by nemo_streaming_parakeet.py) to a StreamingTranslator.

    The translator sees only the words the ASR has committed; a text line extends the open segment,
    an {"is_final": true} line closes it.
    """

    def __init__(self, translator):
        self.translator = translator
        self.words = []
        self.last_end = 0.0

    def feed(self, event, clock, ref):
        if event.get("is_final"):
            self.translator.finalize(" ".join(self.words), self.last_end, clock, ref)
            self.words = []
        elif event.get("text"):
            self.last_end = float(event.get("end", self.last_end))
            self.words += event["text"].split()
            self.translator.update(" ".join(self.words), self.last_end, clock, ref)
