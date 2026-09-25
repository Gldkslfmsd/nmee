#!/usr/bin/env python3
"""LLM backends for the harmful-error annotation.

One interface, several implementations chosen with --backend:

  vllm    local inference with vLLM (fast, batched; needs a GPU)
  einfra  an OpenAI-compatible chat API, e.g. the e-INFRA LLM service (--api-base, --api-key-env)
  dummy   no model at all: returns an annotation of the first few words of each annotated system,
          for testing the pipeline

A backend takes a list of chats (each a list of {"role", "content"} messages) and returns one string
per chat. Add a new backend by subclassing Backend and registering it in BACKENDS.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, fields


@dataclass
class BackendConfig:
    backend: str = "vllm"
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_new_tokens: int = 1024
    temperature: float = 0.0
    batch_size: int = 64            # einfra: parallel requests; vllm: chats per llm.chat() call
    guided_json: bool = True        # constrain the output to the JSON schema where supported
    # vllm
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = None
    dtype: str = "auto"
    # einfra / any OpenAI-compatible API
    api_base: str = None
    api_key_env: str = "LLM_API_KEY"
    timeout: float = 120.0
    retries: int = 3

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
                   help="vllm: chats per generate() call; einfra: parallel requests")
    g.add_argument("--no-guided-json", dest="guided_json", action="store_false",
                   help="do not constrain the output to the JSON schema")
    g.add_argument("--tensor-parallel-size", type=int, default=BackendConfig.tensor_parallel_size)
    g.add_argument("--gpu-memory-utilization", type=float, default=BackendConfig.gpu_memory_utilization)
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--dtype", default=BackendConfig.dtype)
    g.add_argument("--api-base", default=os.environ.get("LLM_API_BASE"),
                   help="base URL of the OpenAI-compatible API (default: $LLM_API_BASE)")
    g.add_argument("--api-key-env", default=BackendConfig.api_key_env,
                   help=f"environment variable with the API key (default: {BackendConfig.api_key_env})")
    g.add_argument("--timeout", type=float, default=BackendConfig.timeout)
    g.add_argument("--retries", type=int, default=BackendConfig.retries)


class Backend:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, chats, schema=None):
        """chats: [[{"role", "content"}, ...], ...] -> one response string per chat."""
        raise NotImplementedError


class VLLMBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        from vllm import LLM
        t0 = time.time()
        kwargs = dict(model=cfg.model, tensor_parallel_size=cfg.tensor_parallel_size,
                      gpu_memory_utilization=cfg.gpu_memory_utilization, dtype=cfg.dtype)
        if cfg.max_model_len:
            kwargs["max_model_len"] = cfg.max_model_len
        self.llm = LLM(**kwargs)
        print(f"vllm: loaded {cfg.model} in {time.time() - t0:.0f}s", file=sys.stderr)

    def _sampling(self, schema):
        from vllm import SamplingParams
        kwargs = dict(temperature=self.cfg.temperature, max_tokens=self.cfg.max_new_tokens)
        if schema and self.cfg.guided_json:
            try:  # the API for constrained decoding has moved between vLLM versions
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
            except ImportError:
                kwargs["guided_json"] = schema
        return SamplingParams(**kwargs)

    def generate(self, chats, schema=None):
        params = self._sampling(schema)
        out = []
        for i in range(0, len(chats), self.cfg.batch_size):
            batch = chats[i:i + self.cfg.batch_size]
            try:
                results = self.llm.chat(batch, params, use_tqdm=False)
            except TypeError:  # older vLLM without use_tqdm
                results = self.llm.chat(batch, params)
            out += [r.outputs[0].text for r in results]
        return out


class OpenAICompatibleBackend(Backend):
    """Any OpenAI-compatible /chat/completions endpoint (e-INFRA, vLLM server, OpenAI itself)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.api_base:
            sys.exit("--api-base (or $LLM_API_BASE) is required for this backend")
        self.key = os.environ.get(cfg.api_key_env, "")
        if not self.key:
            print(f"WARNING: ${cfg.api_key_env} is empty; sending requests without a key",
                  file=sys.stderr)
        self.url = cfg.api_base.rstrip("/") + "/chat/completions"

    def _one(self, chat, schema):
        import urllib.error
        import urllib.request
        body = {"model": self.cfg.model, "messages": chat,
                "temperature": self.cfg.temperature, "max_tokens": self.cfg.max_new_tokens}
        if schema and self.cfg.guided_json:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "annotations", "schema": schema,
                                                       "strict": True}}
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        for attempt in range(self.cfg.retries):
            try:
                req = urllib.request.Request(self.url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as e:
                last = e
                if "response_format" in body and isinstance(e, urllib.error.HTTPError) and e.code == 400:
                    body.pop("response_format")  # server does not support schemas
                    data = json.dumps(body).encode("utf-8")
                time.sleep(2 ** attempt)
        print(f"WARNING: request failed after {self.cfg.retries} attempts: {last}", file=sys.stderr)
        return '{"annotations": []}'

    def generate(self, chats, schema=None):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, self.cfg.batch_size)) as pool:
            return list(pool.map(lambda c: self._one(c, schema), chats))


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


BACKENDS = {"vllm": VLLMBackend, "einfra": OpenAICompatibleBackend, "dummy": DummyBackend}


def get_backend(cfg):
    if cfg.backend not in BACKENDS:
        sys.exit(f"unknown backend {cfg.backend!r} (have: {', '.join(sorted(BACKENDS))})")
    return BACKENDS[cfg.backend](cfg)
