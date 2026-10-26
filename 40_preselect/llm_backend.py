#!/usr/bin/env python3
"""LLM backends for the harmful-error annotation.

One interface, several implementations chosen with --backend:

  vllm    local inference with vLLM (fast, batched; needs a GPU)
  api     any OpenAI-compatible chat API: --api-base is required (e.g. a vLLM server, ÚFAL's
          https://ai.ufal.mff.cuni.cz/v1, OpenAI itself), --model is passed through as given
  einfra   the e-INFRA LLM service: same backend with the base URL and the model short names
          (gpt-oss-120B, GLM, Kimi, DeepSeek, Gemma4, qwen3.8-27b) filled in, and the key taken
          from $E_INFRA_API_TOKEN or $CESNET_API_KEY
  dummy   no model at all: returns an annotation of the first few words of each annotated system,
          for testing the pipeline

A backend takes a list of chats (each a list of {"role", "content"} messages) and returns one string
per chat: generate() returns them all at once, generate_iter() yields them in input order as soon as
they are available. A response of None means the request failed (API backends only).
Add a new backend by subclassing Backend and registering it in BACKENDS.
"""
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, fields


@dataclass
class BackendConfig:
    backend: str = "vllm"
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 4000
    temperature: float = 0.0
    batch_size: int = 1            # einfra: parallel requests; vllm: chats per llm.chat() call
    guided_json: bool = True        # constrain the output to the JSON schema where supported
    # vllm
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    dtype: str = "auto"
    vllm_subprocess: bool = False
    vllm_plugins: bool = False
    vllm_kwargs: str = None         # JSON object with more arguments for vllm.LLM(...)
    # einfra / any OpenAI-compatible API
    api_base: str = None
    api_key_env: str = "E_INFRA_API_TOKEN,CESNET_API_KEY,LLM_API_KEY"
    timeout: float = 120.0
    retries: int = 3
    rate_limit_retries: int = 30    # extra retries on 429 / 503, with backoff, not counted in retries
    max_wait: float = 120.0         # longest wait between two attempts after a 429 / 503

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)})


def add_arguments(parser):
    g = parser.add_argument_group("LLM backend")
    g.add_argument("--backend", choices=sorted(BACKENDS), default="vllm")
    g.add_argument("--model", default=BackendConfig.model,
                   help="model path or ID (vllm), or model name of the API (einfra)")
    g.add_argument("--max-new-tokens", type=int, default=BackendConfig.max_new_tokens)
    g.add_argument("--temperature", type=float, default=BackendConfig.temperature)
    g.add_argument("--batch-size", type=int, default=BackendConfig.batch_size,
                   help="vllm: chats per generate() call (= granularity of the gradual output); "
                        "einfra: parallel requests")
    g.add_argument("--no-guided-json", dest="guided_json", action="store_false",
                   help="do not constrain the output to the JSON schema")
    g.add_argument("--tensor-parallel-size", type=int, default=BackendConfig.tensor_parallel_size)
    g.add_argument("--gpu-memory-utilization", type=float, default=BackendConfig.gpu_memory_utilization)
    g.add_argument("--max-model-len", type=int, default=BackendConfig.max_model_len,
                   help=f"vllm: context length; the default {BackendConfig.max_model_len} is plenty for "
                        f"one segment and keeps the KV cache small (0 = the model's own maximum)")
    g.add_argument("--dtype", default=BackendConfig.dtype)
    g.add_argument("--vllm-subprocess", action="store_true",
                   help="vllm: run the engine in a subprocess (vLLM's default; may fail to init CUDA)")
    g.add_argument("--vllm-plugins", action="store_true",
                   help="vllm: load third-party vLLM plugins (NeMo etc.), off by default")
    g.add_argument("--vllm-kwargs", metavar="JSON",
                   help="vllm: more arguments for vllm.LLM(...) as a JSON object, e.g. "
                        "'{\"kernel_config\": {\"linear_backend\": \"marlin\"}, \"enforce_eager\": true}'; "
                        "they override the options above")
    g.add_argument("--api-base", default=os.environ.get("LLM_API_BASE"),
                   help=f"base URL of the OpenAI-compatible API, usually ending in /v1 "
                        f"(default: $LLM_API_BASE; --backend einfra uses {EINFRA_URL})")
    g.add_argument("--api-key-env", default=BackendConfig.api_key_env,
                   help=f"environment variable(s) with the API key, first non-empty wins "
                        f"(default: {BackendConfig.api_key_env})")
    g.add_argument("--timeout", type=float, default=BackendConfig.timeout)
    g.add_argument("--retries", type=int, default=BackendConfig.retries)
    g.add_argument("--rate-limit-retries", type=int, default=BackendConfig.rate_limit_retries,
                   help="API: how many times to wait and retry a request rejected with 429 Too Many "
                        "Requests or 503, in addition to --retries (the wait follows the server's "
                        "Retry-After, else grows exponentially up to --max-wait)")
    g.add_argument("--max-wait", type=float, default=BackendConfig.max_wait,
                   help="API: longest wait in seconds before retrying after a 429 / 503")


# e-INFRA LLM service (URL and model names as used in 20-find-bad-translation-divergencies.py)
EINFRA_URL = "https://llm.ai.e-infra.cz/v1/"
# short name -> (model name at the API, extra body fields). The "thinking" models (GLM, Kimi,
# DeepSeek) can spend the whole token budget on hidden reasoning, hence reasoning_effort=low.
_LOW_REASONING = {"reasoning_effort": "low"}
EINFRA_MODELS = {
    "Gemma4": ("gemma4", {}),
    "gpt-oss-120B": ("gpt-oss-120b", {}),
    "qwen3.8-27b": ("qwen3.8-27b", {}),
    "GLM": ("glm", _LOW_REASONING),
    "Kimi": ("kimi", _LOW_REASONING),
    "DeepSeek": ("deepseek-v4-flash", _LOW_REASONING),
}


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate_iter(self, chats, schema=None):
        """Yield one response string per chat, in input order, as soon as they are available.
        A response of None means the request failed.
        Default: process in chunks of batch_size via generate()."""
        step = max(1, self.cfg.batch_size)
        for i in range(0, len(chats), step):
            yield from self.generate(chats[i:i + step], schema)

    def generate(self, chats, schema=None):
        """chats: [[{"role", "content"}, ...], ...] -> one response string per chat."""
        raise NotImplementedError


class VLLMBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        # In-process engine by default: vLLM's engine subprocess fails to initialise CUDA in some
        # environments ("CUDA driver initialization failed"), e.g. when `multiprocess` (NeMo) is
        # installed. --vllm-subprocess restores vLLM's own default.
        if not cfg.vllm_subprocess:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # third-party vLLM plugins (NeMo registers one) are not needed here and can fail to import
        if not cfg.vllm_plugins:
            os.environ.setdefault("VLLM_PLUGINS", "")
        from vllm import LLM
        t0 = time.time()
        kwargs = dict(model=cfg.model, tensor_parallel_size=cfg.tensor_parallel_size,
                      gpu_memory_utilization=cfg.gpu_memory_utilization, dtype=cfg.dtype)
        if cfg.max_model_len:
            kwargs["max_model_len"] = cfg.max_model_len
        if cfg.vllm_kwargs:
            try:
                extra = json.loads(cfg.vllm_kwargs)
            except json.JSONDecodeError as e:
                sys.exit(f"--vllm-kwargs is not valid JSON: {e}")
            if not isinstance(extra, dict):
                sys.exit("--vllm-kwargs must be a JSON object, e.g. '{\"enforce_eager\": true}'")
            kwargs.update(extra)
            print(f"vllm: extra arguments {extra}", file=sys.stderr)
        self.llm = LLM(**kwargs)
        print(f"vllm: loaded {cfg.model} in {time.time() - t0:.0f}s", file=sys.stderr)

    def _sampling(self, schema):
        from vllm import SamplingParams
        kwargs = dict(temperature=self.cfg.temperature, max_tokens=self.cfg.max_new_tokens)
        if schema and self.cfg.guided_json:
            # the API for constrained decoding has changed between vLLM versions; try newest first
            try:  # vLLM >= 0.11 / 0.30
                from vllm.sampling_params import StructuredOutputsParams
                kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
            except ImportError:
                try:  # vLLM 0.6 - 0.10
                    from vllm.sampling_params import GuidedDecodingParams
                    kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
                except ImportError:
                    kwargs["guided_json"] = schema
        try:
            return SamplingParams(**kwargs)
        except TypeError as e:  # unknown keyword: fall back to unconstrained decoding
            print(f"vllm: constrained decoding not available ({e}); the prompt alone has to keep the "
                  f"output valid JSON", file=sys.stderr)
            kwargs.pop("structured_outputs", None)
            kwargs.pop("guided_decoding", None)
            kwargs.pop("guided_json", None)
            return SamplingParams(**kwargs)

    def generate_iter(self, chats, schema=None):
        params = self._sampling(schema)
        for i in range(0, len(chats), self.cfg.batch_size):
            batch = chats[i:i + self.cfg.batch_size]
            try:
                results = self.llm.chat(batch, params, use_tqdm=False)
            except TypeError:  # older vLLM without use_tqdm
                results = self.llm.chat(batch, params)
            for r in results:
                yield r.outputs[0].text

    def generate(self, chats, schema=None):
        return list(self.generate_iter(chats, schema))


class APIBackend(Backend):
    """Any OpenAI-compatible /chat/completions endpoint; --api-base is required.
    Subclasses of it only fill in a default URL and a table of model short names."""

    DEFAULT_URL = None
    MODELS = {}

    def __init__(self, cfg):
        super().__init__(cfg)
        base = cfg.api_base or self.DEFAULT_URL
        if not base:
            sys.exit("--api-base (or $LLM_API_BASE) is required for --backend api; "
                     "--backend einfra has the e-INFRA URL built in")
        self.base = base.rstrip("/")
        if not re.search(r"/v\d+$", self.base):
            print(f"note: {self.base} does not end with a version path; OpenAI-compatible servers "
                  f"usually live at .../v1 -- if requests fail with 404/405, try --api-base "
                  f"{self.base}/v1", file=sys.stderr)
        self.url = self.base + "/chat/completions"
        self.model, self.extra = self.MODELS.get(cfg.model, (cfg.model, {}))
        names = [n.strip() for n in cfg.api_key_env.split(",") if n.strip()]
        self.key = next((os.environ[n] for n in names if os.environ.get(n, "").strip()), "")
        if not self.key:
            print(f"WARNING: none of ${', $'.join(names)} is set; sending requests without a key",
                  file=sys.stderr)
        print(f"{cfg.backend}: {self.url}, model {self.model}"
              + (f", {self.extra}" if self.extra else ""), file=sys.stderr)

    def _one(self, chat, schema):
        """The answer of the model, or None if the request failed after all retries."""
        import urllib.error
        import urllib.request
        body = {"model": self.model, "messages": chat, "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_new_tokens, **self.extra}
        if schema and self.cfg.guided_json:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "annotations", "schema": schema,
                                                       "strict": True}}
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        attempt = 0   # failed attempts, without the rate-limited ones
        limited = 0   # 429 / 503 answers of this request
        while attempt < self.cfg.retries:
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
                last = e
                code = e.code if isinstance(e, urllib.error.HTTPError) else None
                if code in (429, 503) and limited < self.cfg.rate_limit_retries:
                    limited += 1
                    self._rate_limited(code)
                    time.sleep(self._wait(e, limited))
                    continue  # does not use up --retries
                if code in (404, 405):
                    break  # wrong URL: retrying will not help
                if "response_format" in body and code == 400:
                    body.pop("response_format")  # server does not support schemas
                    data = json.dumps(body).encode("utf-8")
                time.sleep(2 ** attempt)
                attempt += 1
        hint = ""
        import urllib.error as _ue
        if isinstance(last, _ue.HTTPError):
            if last.code in (404, 405):
                hint = (f" -- wrong --api-base? try {self.base}/v1 ; "
                        f"list the models with: --list-models")
            elif last.code in (401, 403):
                hint = f" -- missing or wrong API key (${self.cfg.api_key_env})"
            elif last.code in (429, 503):
                hint = (f" -- still rate limited after {limited} waits; lower --batch-size "
                        f"(now {self.cfg.batch_size} parallel requests)")
        print(f"WARNING: request failed: {last}{hint}", file=sys.stderr)
        return None

    _lock = threading.Lock()
    _limited_total = 0

    def _rate_limited(self, code):
        """Count the 429 / 503 answers; report the first one and then every 100th."""
        with APIBackend._lock:
            APIBackend._limited_total += 1
            n = APIBackend._limited_total
        if n == 1 or n % 100 == 0:
            print(f"note: the server answered {code} (rate limited) {n}x so far; waiting and retrying. "
                  f"If this keeps happening, lower --batch-size (now {self.cfg.batch_size} parallel "
                  f"requests)", file=sys.stderr, flush=True)

    def _wait(self, error, limited):
        """Seconds to wait after a 429 / 503: the server's Retry-After, else exponential with jitter."""
        retry_after = error.headers.get("Retry-After") if getattr(error, "headers", None) else None
        try:
            wait = float(retry_after)
        except (TypeError, ValueError):
            wait = 2 ** min(limited, 10)
        # jitter, so that the parallel requests do not all come back at the same moment
        return min(self.cfg.max_wait, wait) * (0.5 + random.random())

    def list_models(self):
        """GET {base}/models -- what the server offers."""
        import urllib.request
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        req = urllib.request.Request(self.base + "/models", headers=headers)
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data", payload if isinstance(payload, list) else [])
        return [m.get("id", m) if isinstance(m, dict) else m for m in data]

    def generate_iter(self, chats, schema=None):
        """All requests are submitted at once (batch_size in parallel); the answers are yielded in
        input order as they complete. Closing the generator (Ctrl-C) cancels the queued requests."""
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=max(1, self.cfg.batch_size))
        try:
            futures = [pool.submit(self._one, c, schema) for c in chats]
            for f in futures:
                yield f.result()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def generate(self, chats, schema=None):
        return list(self.generate_iter(chats, schema))


class DummyBackend(Backend):
    """Testing only: no model. Flags the first four words of every annotated system."""

    def generate(self, chats, schema=None):
        out = []
        for chat in chats:
            text = chat[-1]["content"]
            anns = []
            for line in text.splitlines():
                if line.startswith("[") and " to annotate: " in line:
                    system = line[1:line.index("]")]
                    words = line.split(" to annotate: ", 1)[1].split()[:4]
                    if words:
                        anns.append({"system": system, "span": " ".join(words),
                                     "intended": "(dummy)", "harm_types": ["Other"],
                                     "harmfulness": 1, "explanation": "dummy backend",
                                     "error_source": "unknown"})
            out.append(json.dumps({"annotations": anns}))
        return out


class EInfraBackend(APIBackend):
    """The e-INFRA LLM service: the API backend with its URL and model names filled in."""

    DEFAULT_URL = EINFRA_URL
    MODELS = EINFRA_MODELS


BACKENDS = {"vllm": VLLMBackend, "api": APIBackend, "einfra": EInfraBackend, "dummy": DummyBackend}


def get_backend(cfg):
    if cfg.backend not in BACKENDS:
        sys.exit(f"unknown backend {cfg.backend!r} (have: {', '.join(sorted(BACKENDS))})")
    if cfg.backend == "einfra" and cfg.api_base:
        print(f"note: --backend einfra with --api-base {cfg.api_base} (not the e-INFRA URL); "
              f"--backend api is the plain OpenAI-compatible client", file=sys.stderr)
    if cfg.backend == "vllm" and cfg.api_base:
        sys.exit("--backend vllm loads the model locally and ignores --api-base; for a remote "
                 "OpenAI-compatible server use --backend api --api-base ...")
    return BACKENDS[cfg.backend](cfg)