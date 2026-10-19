"""
Shared code for nemo_streaming_parakeet.py, nemo_streaming_llm.py and nemo_streaming_cascade.py.

Sections:
  * generic       : logging, simulated clock, JSONL reading/writing, append-only committer
  * ASR           : NeMo streaming pipeline (nvidia/parakeet-unified-en-0.6b) with latency presets
  * LLM MT        : NeMo's vLLM LLMTranslator driven incrementally (NeMo@IWSLT 2026 cascade)

JSONL format, used for ASR and MT output (SimulStreaming-style, with an unconfirmed tail): one line per
processing step of the simulated system:
    {"emission_time": 8.84, "start": 4.32, "end": 5.76, "text": " And so,",
     "unconfirmed_text": " we will", "is_final": false}
  text              words confirmed in this step (an increment: concatenating "text" gives the output);
                    append-only, never changed later
  unconfirmed_text  the rest of the current hypothesis of the open segment; may change in later steps
  start, end        the source audio that arrived since the previous step, in seconds (from the chunk grid;
                    not the time span of "text": there are no per-word times)
  emission_time     simulated time at which the line is available
  is_final          end of the segment (EoU): "text" holds the remainder, buffers are reset

Only computation-UNAWARE simulation is implemented: emission_time is the audio time of the ASR step (or the
upstream emission time) and ignores the compute time of ASR and LLM. A warning is printed at startup.
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
MAX_NEW_TOKENS = 100  # tokens generated per LLM call (NeMo's default sampling_params.max_tokens)


# --------------------------------------------------------------------------- generic


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Clock:
    """Simulated clock. Only computation-unaware simulation is implemented: emission times are the
    reference times given by the caller (the audio time of the ASR step, or the upstream emission time);
    compute time is not added. (A computation-aware mode needs overlapping steps, not implemented yet.)"""

    def __init__(self):
        self.now = 0.0

    def wait_until(self, t):
        self.now = max(self.now, t)

    @contextlib.contextmanager
    def compute(self):
        yield

    def emission(self, ref):
        return ref

    def end_emission(self, duration):
        return duration


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
    """Writes one line per step; every method also returns the dict it wrote."""

    def __init__(self, fh):
        self.fh = fh

    @staticmethod
    def _join(words):
        return "".join(" " + w for w in words)

    def step(self, new_words, unconfirmed_words, start, end, emission_time, is_final=False):
        obj = {
            "emission_time": round(emission_time, 4),
            "start": round(start, 3),
            "end": round(end, 3),
            "text": self._join(new_words),
            "unconfirmed_text": self._join(unconfirmed_words),
            "is_final": is_final,
        }
        self.fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.fh.flush()
        return obj


class Committer:
    """Turns unstable (partial, final) ASR hypotheses into the JSONL lines described above.

    LocalAgreement-2: a word is confirmed once two consecutive partial hypotheses agree on it; the rest
    of the current hypothesis is the unconfirmed tail. At EoU (non-empty final text) the rest of the final
    text is confirmed and the segment closed. A line is written in every step that has a hypothesis
    (also when nothing changed: a consumer such as the LLM runs one step per ASR step).
    """

    def __init__(self, writer):
        self.writer = writer
        self.committed = []   # words confirmed since the last EoU
        self.prev = []        # previous partial hypothesis (words)
        self.had_tail = False  # the previous line had an unconfirmed tail
        self.revisions = 0    # confirmed words later contradicted by the EoU text
        self.last_end = 0.0   # audio time at the end of the last line

    def _reset(self):
        self.committed, self.prev, self.had_tail = [], [], False

    def step(self, final_text, partial_text, prev_audio_t, audio_t, emission_time):
        """Returns the list of JSONL events (dicts) written by this step."""
        final = final_text.split() if final_text else []
        hyp = partial_text.split() if partial_text else []

        if final:  # end of utterance: confirm the rest, close the segment
            if final[: len(self.committed)] != self.committed:
                self.revisions += 1
            ev = self.writer.step(final[len(self.committed):], [], prev_audio_t, audio_t, emission_time, True)
            self.last_end = audio_t
            self._reset()
            return [ev]

        new = []
        lcp = common_prefix_len(self.prev, hyp)
        if lcp > len(self.committed):
            new = hyp[len(self.committed):lcp]
            self.committed = hyp[:lcp]
        tail = hyp[len(self.committed):]
        self.prev = hyp
        if new or tail or self.had_tail:
            self.had_tail = bool(tail)
            self.last_end = audio_t
            return [self.writer.step(new, tail, prev_audio_t, audio_t, emission_time)]
        return []

    def flush(self, duration, emission_time):
        """End of audio: confirm whatever is pending and close an open segment."""
        if self.prev or self.committed:
            ev = self.writer.step(self.prev[len(self.committed):], [], self.last_end, duration, emission_time, True)
            self._reset()
            return [ev]
        return []


# --------------------------------------------------------------------------- ASR


def warn_comp_unaware():
    log("WARNING: computation-UNAWARE simulation: emission_time ignores the compute time of ASR and LLM "
        "(computation-aware mode is not implemented).")


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
    g.add_argument("--max-num-seqs", type=int, default=8,
                   help="vLLM max_num_seqs (default: %(default)s). Only one request runs at a time here; vLLM's "
                   "default of 1024 makes Qwen3.5 fail with 'max_num_seqs exceeds available Mamba cache blocks' "
                   "and slows the CUDA-graph capture.")
    g.add_argument("--waitk", type=int, default=-1,
                   help="max lag in words between transcript and translation; -1 = longest common prefix "
                   "only (default, as in the paper)")
    g.add_argument("--context-segments", type=int, default=1,
                   help="how many finished segments (source and target side) are given to the LLM as context "
                   "(default: %(default)s = only the last one, as NeMo's streaming pipeline does; 0 = none)")


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
    # vLLM's engine runs in this process (its subprocess failed with "CUDA driver initialization failed" on
    # our cluster nodes), and no third-party vLLM plugins are loaded (NeMo registers one, not needed here).
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_PLUGINS", "")
    # vLLM/FlashInfer compile some kernels at startup with `ninja`; when the script is run as
    # venv/bin/python without activating the venv, a ninja installed in the venv is not on PATH.
    bindir = os.path.dirname(sys.executable)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
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
    sampling_params = {"max_tokens": MAX_NEW_TOKENS, "temperature": 0.0, "top_p": 0.9, "seed": 42}
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


class StreamingTranslator:
    """Incremental LLM translation of a streaming transcript, following NeMo@IWSLT 2026.

    At every source update the LLM translates the current transcript of the open segment, with the
    already fixed translation prefix forced (pre-filled in the assistant turn) and the finished
    segments as context. The next prefix is NeMo's own rule (LLMTranslator.get_prefixes): the longest
    common prefix of two consecutive translations, cut back to a word boundary (and the waitk rule if
    --waitk > 0). That fixed prefix is the confirmed "text" of the output (append-only); the rest of the
    current translation is "unconfirmed_text". At EoU the whole segment is translated, the rest is
    written, and prefix and contexts are updated.

    Prompt length (not handled by NeMo, whose vLLM call fails when the prompt exceeds max_model_len): with
    max_prompt_tokens set, every prompt is kept within that many tokens. If the prompt with the context does
    not fit, the context is left out completely for that call (no partial trimming). If the open segment
    alone, without context, exceeds half of the budget, it is split: the
    current text is translated completely, the rest of that translation is confirmed (without is_final;
    the ASR segment goes on), it becomes the context, and the following updates translate only the source
    words after the split.

    Methods take `clock` (a Clock), `start`/`end` (the source audio interval of the step, copied to the
    output) and `ref` (reference time for emission_time).
    """

    def __init__(self, llm, src_lang, tgt_lang, writer, context_segments=1, max_prompt_tokens=None):
        self.llm = llm
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.writer = writer
        self.context_segments = context_segments
        self.max_prompt_tokens = max_prompt_tokens
        self._tokenizer = None
        self.segments = []           # finished (source, translation) pairs, newest last
        self.src_offset = 0          # source words of the open ASR segment already translated (splits)
        self.last_full = ""          # whole source text of the open ASR segment at the last update
        self._reset_open()
        self.revisions = 0           # written words later contradicted by the final translation
        self.llm_calls = 0
        self.trimmed = 0             # LLM calls without context because it did not fit
        self.splits = 0              # forced splits of over-long segments

    def _reset_open(self):
        self.prefix = ""             # fixed translation prefix of the open (part of the) segment
        self.prev_translation = ""
        self.committed = []          # words already written for it
        self.last_src = ""           # its source text at the last LLM call
        self.last_translation = ""
        self.had_tail = False        # the previous line had an unconfirmed tail

    # ---- prompt length

    def _n_tokens(self, src, prefix, src_ctx, tgt_ctx):
        if self._tokenizer is None:
            self._tokenizer = self.llm.nmt_model.get_tokenizer()
        prompt = self.llm.prompt_template.format(self.src_lang, self.tgt_lang, src, prefix, src_ctx, tgt_ctx)
        return len(self._tokenizer.encode(prompt))

    def _contexts(self, src_text):
        last = self.segments[len(self.segments) - self.context_segments:] if self.context_segments > 0 else []
        src_ctx, tgt_ctx = " ".join(x for x, _ in last), " ".join(y for _, y in last)
        if not self.max_prompt_tokens or not (src_ctx or tgt_ctx):
            return src_ctx, tgt_ctx
        if self._n_tokens(src_text, self.prefix, src_ctx, tgt_ctx) <= self.max_prompt_tokens:
            return src_ctx, tgt_ctx
        self.trimmed += 1  # does not fit: no context at all for this call
        return "", ""

    def _too_long(self, src_text):
        return bool(self.max_prompt_tokens) and \
            self._n_tokens(src_text, self.prefix, "", "") > self.max_prompt_tokens // 2

    # ---- translation steps

    def _translate(self, src_text):
        if src_text == self.last_src and self.last_translation:
            return self.last_translation
        src_ctx, tgt_ctx = self._contexts(src_text)
        self.llm_calls += 1
        return self.llm.translate(
            [src_text], [self.prefix], [self.src_lang], [self.tgt_lang], [src_ctx], [tgt_ctx]
        )[0]

    def _open_text(self, full_text):
        """The part of the ASR segment's text not yet covered by a forced split."""
        return " ".join(full_text.split()[self.src_offset:])

    def update(self, src_text, start, end, clock, ref):
        """New (possibly unstable) source text of the open ASR segment: one LLM step."""
        full = " ".join(src_text.split())
        src_text = self._open_text(full)
        if not src_text:
            return
        self.last_full = full
        if self._too_long(src_text):
            self._close(src_text, start, end, clock, ref, is_final=False)
            self.src_offset = len(full.split())
            self.splits += 1
            return
        # An unchanged source is not skipped: NeMo runs one LLM step per ASR step, and the next step on
        # the same source gives the same translation, so the common prefix with the previous one is the
        # whole translation and the fixed prefix advances (_translate reuses the translation, no LLM call).
        with clock.compute():
            translation = self._translate(src_text)
            new_prefix = self.llm.get_prefixes([src_text], [translation], [self.prev_translation])[0]
        self.last_src, self.last_translation = src_text, translation
        self.prev_translation = translation
        words = new_prefix.split()
        new = []
        if len(words) > len(self.committed) and words[: len(self.committed)] == self.committed:
            new = words[len(self.committed):]
            self.committed = words
        t_words = translation.split()
        tail = t_words[len(self.committed):] if t_words[: len(self.committed)] == self.committed else []
        if new or tail or self.had_tail:
            self.writer.step(new, tail, start, end, clock.emission(ref))
            self.had_tail = bool(tail)
        self.prefix = new_prefix

    def _close(self, src_text, start, end, clock, ref, is_final):
        """Translate src_text completely, write the rest, make it context, reset the open part."""
        translation = ""
        if src_text:
            with clock.compute():
                translation = self._translate(src_text)
        words = translation.split()
        if words[: len(self.committed)] != self.committed:
            self.revisions += 1
        if words[len(self.committed):] or is_final:
            self.writer.step(words[len(self.committed):], [], start, end, clock.emission(ref), is_final)
        if src_text:
            self.segments.append((src_text, translation.strip()))
            self.segments = self.segments[-50:]
        self._reset_open()

    def finalize(self, src_text, start, end, clock, ref):
        """The ASR segment is complete (EoU): translate the rest fully, write it, reset."""
        full = " ".join(src_text.split())
        if not full:
            return
        self._close(self._open_text(full), start, end, clock, ref, is_final=True)
        self.src_offset = 0
        self.last_full = ""

    def finish(self, start, end, clock, ref):
        """End of input: close a segment that never got an EoU."""
        if self.last_full:
            self.finalize(self.last_full, start, end, clock, ref)


class AsrEventFeeder:
    """Feeds ASR JSONL lines (as written by nemo_streaming_parakeet.py) to a StreamingTranslator,
    one LLM step per line.

    With use_unconfirmed (default) the source of a step is the confirmed text of the open segment plus
    "unconfirmed_text", i.e. the current ASR hypothesis, as in NeMo; otherwise only the confirmed text.
    An is_final line closes the segment: its final transcript is the confirmed text plus that line's "text".
    (Files from the older format, without unconfirmed_text and with a separate is_final line, also work.)
    """

    def __init__(self, translator, use_unconfirmed=True):
        self.translator = translator
        self.use_unconfirmed = use_unconfirmed
        self.words = []       # confirmed words of the open segment
        self.last_start = 0.0
        self.last_end = 0.0

    def feed(self, event, clock, ref):
        self.last_start = float(event.get("start", self.last_end))
        self.last_end = float(event.get("end", self.last_end))
        new = (event.get("text") or "").split()
        if event.get("is_final"):
            self.translator.finalize(" ".join(self.words + new), self.last_start, self.last_end, clock, ref)
            self.words = []
            return
        self.words += new
        tail = (event.get("unconfirmed_text") or "").split() if self.use_unconfirmed else []
        if self.words or tail:
            self.translator.update(" ".join(self.words + tail), self.last_start, self.last_end, clock, ref)


def add_mt_input_arg(ap):
    ap.add_argument("--mt-input", choices=["partial", "committed"], default="partial",
                    help="what the LLM translates: the current ASR hypothesis, i.e. confirmed plus unconfirmed "
                    "text (default, as in NeMo and the paper), or only the confirmed ASR text")