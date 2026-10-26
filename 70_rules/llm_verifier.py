#!/usr/bin/env python3
"""LLM verifier of verification rules: does a translation pass or fail a rule?

Used by rule_editor.py (interactively, one span at a time) and runnable on its own (batch):

  python llm_verifier.py --systems systems.jsonl --rules rules_verified.jsonl -o verified.jsonl \
      --backend einfra --model gpt-oss-120B --reuse-editor-cache

For every span in --rules (the output of rule_editor.py; by default only the ones marked finished),
every rule is checked against the translations of all systems of that segment in the target
language (from --systems) and against the span's contrastive translations. A translation is
harmful if it fails at least one rule. Writes one JSON line per span to --output and prints a
summary (harmful spans per system, LLM calls, problems).

LLM answers are cached on disk (default <output>.llm-cache.jsonl), so a second run asks only about
new (translation, rule) pairs. --reuse-editor-cache also reads the cache of rule_editor.py
(<rules>.llm-cache.jsonl, read-only): every verdict the editor got with the same backend, model,
settings and prompt is reused instead of asking the LLM again.

The backend options (--backend, --model, --api-base, ...) come from llm_backend.py.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import threading
import time

import llm_backend

DEFAULT_PROMPT = """Your goal is to verify whether a translation fulfills a criterion.
Criterion: {rule}
Input: {source_text}
Translation to verify: {translation}
Output only pass or fail and nothing else."""

PLACEHOLDERS = ("rule", "source_text", "translation")


def fill_prompt(template, source, translation, rule):
    """The prompt with {rule}, {source_text} and {translation} filled in (in one pass, so that braces
    in the texts are left alone). {source} is accepted as an older name of {source_text}."""
    values = {"rule": rule, "source_text": source, "source": source, "translation": translation}
    return re.sub(r"\{(rule|source_text|source|translation)\}", lambda m: values[m.group(1)], template)


def parse_verdict(text):
    """{'verdict': 'pass'|'fail'|'error', 'evidence': ...} from a model answer.
    The prompt asks for a bare 'pass' or 'fail'; JSON ({"verdict": ...}) and a verdict word inside a
    longer answer are accepted too (then the whole answer is kept as 'raw')."""
    if text is None:
        return {"verdict": "error", "evidence": "request failed"}
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    bare = t.strip(" \t\n.!*`'\"").lower()
    if bare in ("pass", "fail"):
        return {"verdict": bare, "evidence": ""}
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", t, flags=re.S))):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        v = str(o.get("verdict", "")).strip().lower()
        if v in ("pass", "fail"):
            return {"verdict": v, "evidence": str(o.get("evidence", ""))}
    found = re.findall(r"\b(pass|fail)\b", t, flags=re.I)
    if found:  # the last one is the conclusion
        return {"verdict": found[-1].lower(), "evidence": "", "raw": text}
    return {"verdict": "error", "evidence": "unparsable answer", "raw": text}


# ---------------------------------------------------------------------------- systems input

def seg_key(d):
    """Identifies a segment across files: the document and the segment's audio file (or times)."""
    a = os.path.basename(d.get("audio") or "")
    return d.get("document", ""), a or f"{d.get('beg')}-{d.get('end')}"


def load_systems(path):
    """{seg_key: line} from a systems JSONL ({"document", "audio", "systems_info": {system: {"lan"}},
    "text": {system: text}} per segment)."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[seg_key(d)] = d
    return out


def target_texts(sysline, ref_system, lan=None, skip=("gold_transcript",)):
    """{system: text} of all systems of a systems line in the language of ref_system (or lan),
    the reference system included. Systems without a language in systems_info are taken unless
    they are in skip."""
    info = sysline.get("systems_info", {})
    lan = lan or info.get(ref_system, {}).get("lan")
    out = {}
    for s, t in sysline.get("text", {}).items():
        if s == ref_system:
            out[s] = t
        elif s in info and info[s].get("lan") and lan:
            if info[s]["lan"] == lan:
                out[s] = t
        elif s not in skip:
            out[s] = t
    return out


# ---------------------------------------------------------------------------- errors

class VerifyError(Exception):
    """The LLM could not be used; the message is meant for the user."""


class Tee(io.TextIOBase):
    """Write to the real stderr and keep a copy (the backends report failures on stderr)."""

    def __init__(self, real):
        self.real, self.parts = real, []

    def write(self, text):
        self.parts.append(text)
        return self.real.write(text)

    def flush(self):
        self.real.flush()

    def text(self):
        return "".join(self.parts).strip()


def describe(e):
    if isinstance(e, SystemExit):  # the backends call sys.exit("message") on configuration errors
        return str(e.code) if e.code not in (None, 0, 1) else "the backend exited"
    return f"{type(e).__name__}: {e}"


def summarize_reasons(reasons):
    """Distinct failure reasons, each with how many requests it hit."""
    counts = {}
    for r in reasons:
        counts[r] = counts.get(r, 0) + 1
    return "\n".join(f"{n}x {r}" if n > 1 else r for r, n in counts.items())


def _one_safe(backend, chat, schema, times=None, i=None):
    """One request through the API backend; an exception becomes None (= failed) plus the warning
    that the API backend prints for its own failures. times: a dict that gets times[i] = seconds
    the request took (including the backend's retries)."""
    t0 = time.time()
    try:
        return backend._one(chat, schema)
    except Exception as e:  # noqa: BLE001 -- anything from the backend
        print(f"WARNING: request failed: {describe(e)}", file=sys.stderr)
        return None
    finally:
        if times is not None:
            times[i] = time.time() - t0


def generate_unordered(backend, chats, schema=None, times=None):
    """Yield (index, answer) for every chat as soon as it is answered (None = failed).
    times: a dict that gets times[index] = seconds per request (API backends only; with
    --batch-size > 1 this includes waiting at the server while it serves the other requests).

    API backends (with _one): batch_size requests are kept running all the time; when one is
    answered, the next one is sent, so a slow answer (e.g. a long reasoning) holds up only its own
    slot. Exceptions (e.g. ConnectionResetError, which llm_backend does not catch) fail just that
    request. Other backends (vLLM: batched on the GPU anyway) go through their generate_iter in
    chunks of batch_size; a chunk that raises is redone one chat at a time."""
    n_par = max(1, getattr(backend.cfg, "batch_size", 64) or 64)
    if hasattr(backend, "_one"):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pool = ThreadPoolExecutor(max_workers=n_par)
        try:
            futures = {pool.submit(_one_safe, backend, c, schema, times, i): i for i, c in enumerate(chats)}
            for f in as_completed(futures):
                yield futures[f], f.result()
        finally:  # also when the caller stops early (Ctrl-C): drop the requests not sent yet
            pool.shutdown(wait=False, cancel_futures=True)
        return
    for start in range(0, len(chats), n_par):
        chunk = chats[start:start + n_par]
        got = 0
        try:
            for a in backend.generate_iter(chunk, schema):
                yield start + got, a
                got += 1
        except Exception as e:  # noqa: BLE001
            print(f"note: {describe(e)} in a batch of {len(chunk)} requests; asking one by one",
                  file=sys.stderr)
            for k in range(got, len(chunk)):
                try:
                    yield start + k, backend.generate([chunk[k]], schema)[0]
                except Exception as e1:  # noqa: BLE001
                    print(f"WARNING: request failed: {describe(e1)}", file=sys.stderr)
                    yield start + k, None


def safe_generate_iter(backend, chats, schema=None):
    """The answers in input order (None = failed), from generate_unordered(): all requests keep
    running in parallel; an answer is passed on as soon as all answers before it are there."""
    buf, nxt = {}, 0
    for i, a in generate_unordered(backend, chats, schema):
        buf[i] = a
        while nxt in buf:
            yield buf.pop(nxt)
            nxt += 1


# ---------------------------------------------------------------------------- cache

class Cache:
    """Disk cache of LLM answers: one JSON line per answer, appended as they come.
    The key is a hash of everything that determines the answer: backend, model, decoding settings
    and the full chat (the prompt with rule, source and translation filled in). Only answers with a
    pass/fail verdict are stored, so failed or unparsable requests are asked again next time.
    path: read and written; reuse: more cache files that are only read (e.g. the editor's)."""

    def __init__(self, path, reuse=()):
        self.path, self.data, self.lock = path, {}, threading.Lock()
        self.secs = {}   # key -> seconds the request took, when known
        for p in [*reuse, path]:
            if p and os.path.exists(p):
                n = len(self.data)
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            self.data[r["key"]] = r["answer"]
                            if r.get("seconds") is not None:
                                self.secs[r["key"]] = r["seconds"]
                        except (json.JSONDecodeError, KeyError):
                            pass  # a line cut off by a crash
                print(f"cache: {len(self.data) - n} LLM answers from {p}"
                      + ("" if p == path else " (read only)"), file=sys.stderr)
            elif p and p != path:
                print(f"WARNING: cache to reuse not found: {p}", file=sys.stderr)

    @staticmethod
    def key(settings, chat):
        return hashlib.sha256(json.dumps([settings, chat], ensure_ascii=False, sort_keys=True)
                              .encode("utf-8")).hexdigest()

    def get(self, key):
        return self.data.get(key)

    def put_many(self, entries, model):
        """entries: [(key, answer)] or [(key, answer, seconds)]"""
        if not entries:
            return
        entries = [(e[0], e[1], e[2] if len(e) > 2 else None) for e in entries]
        with self.lock:
            for k, a, sec in entries:
                self.data[k] = a
                if sec is not None:
                    self.secs[k] = sec
            if not self.path:
                return
            with open(self.path, "a", encoding="utf-8") as f:
                for k, a, sec in entries:
                    r = {"key": k, "model": model, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "answer": a}
                    if sec is not None:
                        r["seconds"] = round(sec, 3)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------- extra request fields

THINKING_LEVELS = ("no", "low", "medium", "high")
DEFAULT = object()   # Verifier.answer(thinking=DEFAULT): the level given to the Verifier


def add_thinking_argument(parser):
    parser.add_argument("--enable-thinking", choices=THINKING_LEVELS, default=None,
                        help="reasoning of the model (API backends): no = off, or the effort low / medium / "
                             "high. Sent in the forms the common models read (Qwen: enable_thinking; "
                             "gpt-oss, Qwen3.8: reasoning_effort); a server that knows none of them "
                             "ignores it. Default: nothing is sent, the model decides. Part of the cache key")


def thinking_body(level):
    """The request fields for --enable-thinking (an empty dict if not given)."""
    if not level:
        return {}
    if level == "no":
        # no top-level reasoning_effort here: vLLM accepts only low/medium/high there
        return {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "none"}}
    return {"reasoning_effort": level,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": level}}


def apply_extra_body(backend, extra):
    """Add the fields to every request of an llm_backend API backend (its 'extra' dict)."""
    if not extra:
        return
    if hasattr(backend, "extra"):
        backend.extra = {**(backend.extra or {}), **extra}
        print(f"extra request fields: {json.dumps(extra)}", file=sys.stderr)
    else:
        print(f"WARNING: --enable-thinking works only with the API backends (api, einfra); ignored for "
              f"{type(backend).__name__}", file=sys.stderr)


# ---------------------------------------------------------------------------- progress

class ProgressLog:
    """Progress of a long LLM run on stderr: called with (done, total, failed) after every answer
    (and with done=0 before the first); prints a line every `every` seconds or 10 %, and at the end."""

    def __init__(self, label="LLM", every=10.0):
        self.label, self.every = label, every
        self.t0 = self.last_t = time.time()
        self.last_pct = -1

    def __call__(self, done, total, failed=0):
        now = time.time()
        pct = 100 * done // total if total else 100
        if done == 0:
            print(f"  {self.label}: asking {total} pairs ...", file=sys.stderr, flush=True)
            self.t0 = self.last_t = now
            return
        if done < total and now - self.last_t < self.every and pct // 10 == self.last_pct // 10:
            return
        el = now - self.t0
        rate = done / el if el > 0 else 0
        eta = (total - done) / rate if rate else 0
        print(f"  {self.label}: {done}/{total} ({pct}%), {el:.0f}s, {rate:.1f}/s"
              + (f", ~{eta:.0f}s left" if done < total else ", done")
              + (f", {failed} failed" if failed else ""), file=sys.stderr, flush=True)
        self.last_t, self.last_pct = now, pct


# ---------------------------------------------------------------------------- verifier

class Verifier:
    """cfg: llm_backend.BackendConfig. prompt: a template with {rule}, {source_text}, {translation}
    (default DEFAULT_PROMPT). cache_path: cache file (None = no disk cache); reuse: read-only caches."""

    def __init__(self, cfg, prompt=None, cache_path=None, reuse=(), backend_module=llm_backend, thinking=None):
        self.prompt = prompt or DEFAULT_PROMPT
        missing = [p for p in PLACEHOLDERS if "{%s}" % p not in self.prompt
                   and not (p == "source_text" and "{source}" in self.prompt)]
        if missing:
            print(f"WARNING: the prompt has no {', '.join('{%s}' % p for p in missing)}", file=sys.stderr)
        self.mod, self.cfg = backend_module, cfg
        self.dummy = cfg.backend == "dummy"
        self.backend, self.init_error = None, None
        self.lock = threading.Lock()   # vLLM is not thread-safe; one LLM call at a time
        self.model = "dummy" if self.dummy else f"{cfg.backend}:{cfg.model}"
        # what changes the answer, besides the chat itself
        self.base_settings = {"backend": cfg.backend, "model": cfg.model,
                              "temperature": cfg.temperature, "max_new_tokens": cfg.max_new_tokens}
        self.thinking = thinking       # the default --enable-thinking level (None = send nothing)
        self.base_extra = {}           # the backend's own extra request fields (e.g. e-INFRA's)
        self.settings = self.settings_for(thinking)[0]
        self.cache = Cache(None if self.dummy else cache_path, () if self.dummy else reuse)
        if not self.dummy:
            self._init_backend()

    def _init_backend(self):
        """Load the backend; on failure remember the error (answer() tries again)."""
        tee = Tee(sys.stderr)
        try:
            with contextlib.redirect_stderr(tee):
                self.backend = self.mod.get_backend(self.cfg)
                self.base_extra = dict(getattr(self.backend, "extra", None) or {})
                if self.thinking:
                    apply_extra_body(self.backend, thinking_body(self.thinking))
            self.init_error = None
        except (Exception, SystemExit) as e:
            self.backend = None
            log = tee.text()
            self.init_error = (f"Could not start the AI judge (backend {self.model}): {describe(e)}"
                               + (f"\n\n{log}" if log else ""))
            print(self.init_error, file=sys.stderr)

    def chat(self, source, translation, rule):
        return [{"role": "user", "content": fill_prompt(self.prompt, source, translation, rule)}]

    def settings_for(self, thinking):
        """(cache-key settings, extra request fields) for a thinking level. Without a level the
        settings have no extra_body, so the keys of answers from before the option stay valid."""
        extra = thinking_body(thinking)
        settings = dict(self.base_settings)
        if extra:
            settings["extra_body"] = extra
        return settings, extra

    def label(self, thinking=DEFAULT):
        """The model name, with the thinking level if one is set (for 'verified_with')."""
        level = self.thinking if thinking is DEFAULT else thinking
        return self.model + (f" thinking={level}" if level and not self.dummy else "")

    def answer(self, chats, use_cache=True, progress=None, thinking=DEFAULT, on_answer=None):
        """(answers, stats, reasons): one answer string per chat (None = the request failed),
        {"cached": n, "asked": m, "from_cache": [bool per chat], "seconds": [seconds per chat]},
        and the failure messages of the backend. "seconds" is the time of the request (for a cached
        answer: of the request that produced it, if the cache knows it; else None).
        thinking: a level of --enable-thinking for these requests (DEFAULT: the Verifier's).
        on_answer(index, answer): called as soon as an answer is known: at once for the cached ones,
        then as the backend delivers them (one by one for API backends, per batch for vLLM).
        Raises VerifyError if the backend cannot be used or every request failed."""
        level = self.thinking if thinking is DEFAULT else thinking
        settings, extra = self.settings_for(level)
        keys = [Cache.key(settings, c) for c in chats]
        answers = [self.cache.get(k) if use_cache else None for k in keys]
        todo = [i for i, a in enumerate(answers) if a is None]
        stats = {"cached": len(chats) - len(todo), "asked": len(todo),
                 "from_cache": [a is not None for a in answers],
                 "seconds": [self.cache.secs.get(k) if a is not None else None for k, a in zip(keys, answers)]}
        reasons = []
        if on_answer:
            for i, a in enumerate(answers):
                if a is not None:
                    on_answer(i, a)
        if progress:
            print(f"  {stats['cached']} of {len(chats)} pairs from the cache"
                  + (", nothing to ask" if not todo else ""), file=sys.stderr, flush=True)
        if not todo:
            return answers, stats, reasons
        if progress:
            progress(0, len(todo), 0)
        with self.lock:
            if self.backend is None:
                self._init_backend()
                if self.backend is None:
                    raise VerifyError(self.init_error)
            if hasattr(self.backend, "extra"):  # the level of these requests (the lock is held)
                self.backend.extra = {**self.base_extra, **extra}
            tee = Tee(sys.stderr)
            new, pending, done, n_failed, times = [None] * len(todo), [], 0, 0, {}
            try:
                with contextlib.redirect_stderr(tee):
                    for j, a in generate_unordered(self.backend, [chats[i] for i in todo], None, times):
                        new[j], done = a, done + 1
                        n_failed += a is None
                        stats["seconds"][todo[j]] = times.get(j)
                        if a is not None and parse_verdict(a)["verdict"] != "error":
                            pending.append((keys[todo[j]], a, times.get(j)))
                        if pending:   # write every answer at once: stopping the run keeps all of them
                            self.cache.put_many(pending, self.model)
                            pending = []
                        if on_answer:
                            on_answer(todo[j], a)
                        if progress:
                            progress(done, len(todo), n_failed)
            except (Exception, SystemExit) as e:
                self.cache.put_many(pending, self.model)
                log = tee.text()
                raise VerifyError(f"The AI judge call failed: {describe(e)}" + (f"\n\n{log}" if log else ""))
            self.cache.put_many(pending, self.model)
        log = tee.text()
        # the API backends return None for a request that failed and print why on stderr
        # (parallel requests can interleave their lines, hence the regex)
        reasons = [r.strip() for r in re.findall(r"WARNING: request failed: (.*?)(?=WARNING:|\n|$)", log)]
        failed = sum(a is None for a in new)
        if failed == len(chats):
            raise VerifyError(f"All {failed} AI judge requests failed.\n\n"
                              + (summarize_reasons(reasons) or log or "no details from the backend"))
        for i, a in zip(todo, new):
            answers[i] = a
        return answers, stats, reasons

    def verify(self, source, targets, rules, use_cache=True, dummy_fail=(), progress=None, thinking=DEFAULT,
               on_verdict=None):
        """({target: [verdict per rule]}, [error messages], stats) for {target: translation}.
        dummy_fail: the targets the dummy backend fails (it passes all others).
        on_verdict(target, rule index, verdict): called for every verdict as soon as it is known."""
        jobs = [(s, k) for s in targets for k in range(len(rules))]

        def tell(i, a):
            if on_verdict:
                v = parse_verdict(a)
                if a is None:
                    v["evidence"] = "request failed"
                on_verdict(jobs[i][0], jobs[i][1], v)

        if self.dummy:
            answers = ["fail" if s in dummy_fail else "pass" for s, _ in jobs]
            stats, reasons = {"cached": 0, "asked": len(jobs)}, []
            for i, a in enumerate(answers):
                tell(i, a)
        else:
            chats = [self.chat(source, targets[s], rules[k]) for s, k in jobs]
            answers, stats, reasons = self.answer(chats, use_cache, progress, thinking, on_answer=tell)
        res = {s: [None] * len(rules) for s in targets}
        for (s, k), a in zip(jobs, answers):
            v = parse_verdict(a)
            if a is None:
                v["evidence"] = reasons[0] if reasons else "request failed"
            res[s][k] = v
        return res, verdict_errors(res, len(jobs), reasons), stats


def verdict_errors(res, n_jobs, reasons=()):
    """Messages about the verdicts that are errors (? in the editor)."""
    errors = []
    vs = [v for row in res.values() for v in row]
    n_failed = sum(v["verdict"] == "error" and v["evidence"] != "unparsable answer" for v in vs)
    unparsed = [v for v in vs if v["verdict"] == "error" and v["evidence"] == "unparsable answer"]
    if n_failed:
        errors.append(f"{n_failed} of {n_jobs} AI judge requests failed (marked ?)."
                      + (f"\n{summarize_reasons(reasons)}" if reasons else ""))
    if unparsed:
        errors.append(f"{len(unparsed)} AI judge answer(s) had no pass/fail verdict (marked ?). "
                      f"First one:\n{(unparsed[0].get('raw') or '')[:1500]}")
    return errors


def read_prompt(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------- batch

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def records_from_rule_lines(rules_path, data_path, systems):
    """Span records like rule_editor.py writes them, from a synthesize_rules.py file (one rule per
    line) and the confirmed spans, so that synthetic rules can be verified without the editor."""
    if not data_path:
        sys.exit(f"{rules_path} has one rule per line (synthesize_rules.py); give the spans with --data")
    from rule_editor import load_items, load_prefill   # here: rule_editor imports this module
    items = load_items(data_path, systems)
    prefill = load_prefill(rules_path, items)
    print(f"{sum(len(v) for v in prefill.values())} rules for {len(prefill)} of {len(items)} spans "
          f"from {rules_path}", file=sys.stderr)
    return [{"key": it["key"], "document": it["document"], "audio": it["audio"], "seg": it["seg"],
             "span": it["span"], "ref_system": it["ref_system"], "source": it["source"],
             "rules": prefill[it["key"]], "contrastive": [], "decision": None}
            for it in items if prefill.get(it["key"])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", required=True, help="JSONL with the translations of all systems per segment")
    p.add_argument("--rules", required=True, help="the output of rule_editor.py (rules per span), or rules "
                                                  "as written by synthesize_rules.py (one per line; needs --data)")
    p.add_argument("--data", help="the confirmed harmful spans (JSONL, as for rule_editor.py); needed when "
                                  "--rules is in the synthesize_rules.py format")
    p.add_argument("--output", "-o", required=True, help="JSONL with the verdicts, one line per span")
    p.add_argument("--decisions", default="finished",
                   help="which spans to verify, by the annotator's decision: a comma-separated list of "
                        "finished, undecidable, remove, progress (= not decided), or 'all' "
                        "(every span with rules, except remove). Default: finished")
    p.add_argument("--no-contrastive", action="store_true", help="do not verify the contrastive translations")
    p.add_argument("--prompt", help="prompt file with {rule}, {source_text}, {translation} "
                                    "(default: the built-in prompt, the same as in rule_editor.py)")
    p.add_argument("--cache", help="disk cache of LLM answers, read and written (default: <output>.llm-cache.jsonl)")
    p.add_argument("--no-cache", action="store_true", help="neither read nor write the cache; ask the LLM for everything")
    p.add_argument("--reuse-editor-cache", action="store_true",
                   help="also read the cache of rule_editor.py, <rules>.llm-cache.jsonl (never written)")
    p.add_argument("--reuse-cache", action="append", default=[], metavar="FILE",
                   help="also read this cache file (repeatable; never written)")
    add_thinking_argument(p)
    llm_backend.add_arguments(p)
    args = p.parse_args()

    wanted = {"finished", "undecidable", "progress"} if args.decisions == "all" else \
        {d.strip() for d in args.decisions.split(",") if d.strip()}
    bad = wanted - {"finished", "undecidable", "remove", "progress"}
    if bad:
        sys.exit(f"--decisions: unknown {', '.join(sorted(bad))}")

    systems = load_systems(args.systems)
    records = load_jsonl(args.rules)
    if records and "rule" in records[0] and "rules" not in records[0]:
        records = records_from_rule_lines(args.rules, args.data, systems)
    reuse = list(args.reuse_cache)
    if args.reuse_editor_cache:
        reuse.append(os.path.splitext(args.rules)[0] + ".llm-cache.jsonl")
    cache_path = None if args.no_cache else (args.cache or os.path.splitext(args.output)[0] + ".llm-cache.jsonl")
    verifier = Verifier(llm_backend.BackendConfig.from_args(args), read_prompt(args.prompt),
                        cache_path, [] if args.no_cache else reuse, thinking=args.enable_thinking)

    # ---- which spans, which translations
    skipped = {"decision": 0, "no rules": 0, "not in --systems": 0, "reference not in --systems": 0}
    spans = []
    for r in records:
        dec = r.get("decision") or "progress"
        if dec not in wanted:
            skipped["decision"] += 1
            continue
        if not r.get("rules"):
            skipped["no rules"] += 1
            continue
        sl = systems.get(seg_key(r))
        if sl is None:
            skipped["not in --systems"] += 1
            print(f"WARNING: {r['key']}: segment not in --systems, skipped", file=sys.stderr)
            continue
        if r["ref_system"] not in sl.get("text", {}):
            skipped["reference not in --systems"] += 1
            print(f"WARNING: {r['key']}: {r['ref_system']} not in --systems, skipped", file=sys.stderr)
            continue
        targets = target_texts(sl, r["ref_system"])
        contrastive = []
        if not args.no_contrastive:
            for c in r.get("contrastive") or []:
                c = c if isinstance(c, dict) else {"label": "", "text": c}
                if not c.get("text", "").strip():
                    continue
                label = c.get("label") or f"contrastive {len(contrastive) + 1}"
                while label in targets:  # a system of the new --systems has the same name
                    label += " (contrastive)"
                targets[label] = c["text"]
                contrastive.append(label)
        source = sl.get("text", {}).get("gold_transcript") or r.get("source", "")
        spans.append((r, source, targets, contrastive))
    print(f"{len(spans)} spans to verify; skipped: "
          + ", ".join(f"{n} {why}" for why, n in skipped.items() if n) if any(skipped.values()) else
          f"{len(spans)} spans to verify", file=sys.stderr)

    # ---- one LLM call for everything (the backend batches / parallelizes it)
    jobs = [(si, s, k) for si, (r, src, tg, _) in enumerate(spans) for s in tg for k in range(len(r["rules"]))]
    progress = ProgressLog("LLM")

    if verifier.dummy:
        answers = ["fail" if s == spans[si][0]["ref_system"] else "pass" for si, s, _ in jobs]
        stats, reasons = {"cached": 0, "asked": len(jobs)}, []
    else:
        chats = [verifier.chat(spans[si][1], spans[si][2][s], spans[si][0]["rules"][k]) for si, s, k in jobs]
        try:
            answers, stats, reasons = verifier.answer(chats, use_cache=not args.no_cache, progress=progress)
        except VerifyError as e:
            sys.exit(f"ERROR: {e}")

    results = [{s: [None] * len(r["rules"]) for s in tg} for r, _, tg, _ in spans]
    secs, cached = stats.get("seconds") or [None] * len(jobs), stats.get("from_cache") or [False] * len(jobs)
    for j, ((si, s, k), a) in enumerate(zip(jobs, answers)):
        v = parse_verdict(a)
        if a is None:
            v["evidence"] = reasons[0] if reasons else "request failed"
        v["seconds"] = None if secs[j] is None else round(secs[j], 3)   # time of the LLM request
        v["cached"] = cached[j]
        results[si][s][k] = v

    # ---- output
    with open(args.output, "w", encoding="utf-8") as f:
        for (r, src, tg, contr), res in zip(spans, results):
            harmful = {s: any(v["verdict"] == "fail" for v in vs) for s, vs in res.items()}
            f.write(json.dumps({
                "key": r["key"], "document": r.get("document"), "audio": r.get("audio"), "seg": r.get("seg"),
                "span": r.get("span"), "ref_system": r["ref_system"], "decision": r.get("decision"),
                "source": src, "rules": r["rules"], "translations": tg, "contrastive": contr,
                "results": res, "harmful": harmful,
                "ref_fails_each_rule": [v["verdict"] == "fail" for v in res[r["ref_system"]]],
                "verified_with": verifier.model,
            }, ensure_ascii=False) + "\n")

    # ---- summary
    print_summary(spans, results, stats, reasons, args.output, verifier.model)


def print_summary(spans, results, stats, reasons, output, model):
    out = sys.stdout
    n_pairs = sum(len(vs) for res in results for vs in res.values())
    errors = [v for res in results for vs in res.values() for v in vs if v["verdict"] == "error"]
    print(f"\n== {len(spans)} spans verified with {model}; {n_pairs} (translation, rule) pairs: "
          f"{stats['cached']} from cache, {stats['asked']} asked the LLM, {len(errors)} errors", file=out)
    if reasons:
        print("LLM failures:\n  " + summarize_reasons(reasons).replace("\n", "\n  "), file=out)

    # per system: in how many spans is it harmful (fails at least one rule)?
    table, order = {}, []
    for (r, _, tg, contr), res in zip(spans, results):
        for s, vs in res.items():
            name = "(contrastive)" if s in contr else s
            if name not in table:
                table[name] = {"spans": 0, "harmful": 0, "unclear": 0, "ref": 0}
                order.append(name)
            t = table[name]
            t["spans"] += 1
            t["ref"] += s == r["ref_system"]
            if any(v["verdict"] == "fail" for v in vs):
                t["harmful"] += 1
            elif any(v["verdict"] == "error" for v in vs):
                t["unclear"] += 1
    order.sort(key=lambda n: (-table[n]["ref"], n == "(contrastive)", n))
    w = max([len(n) for n in order] + [6])
    print(f"\n{'system':<{w}}  spans  harmful      ok  unclear", file=out)
    for n in order:
        t = table[n]
        ok = t["spans"] - t["harmful"] - t["unclear"]
        pct = 100 * t["harmful"] / t["spans"] if t["spans"] else 0
        print(f"{n:<{w}}  {t['spans']:5d}  {t['harmful']:4d} {pct:3.0f}%  {ok:6d}  {t['unclear']:7d}"
              + ("   (reference: should be harmful everywhere)" if t["ref"] else ""), file=out)

    # problems worth a look
    ok = lambda vs: all(v["verdict"] == "pass" for v in vs)   # noqa: E731 -- overall pass
    ref_passes = [r["key"] for (r, *_), res in zip(spans, results) if ok(res[r["ref_system"]])]
    nobody = [r["key"] for (r, *_), res in zip(spans, results) if not any(ok(vs) for vs in res.values())]
    if ref_passes:
        print(f"\n{len(ref_passes)} span(s) where the reference passes all rules (they do not catch the "
              f"confirmed error): " + ", ".join(ref_passes[:10]) + (" ..." if len(ref_passes) > 10 else ""), file=out)
    if nobody:
        print(f"{len(nobody)} span(s) where no translation passes all rules: "
              + ", ".join(nobody[:10]) + (" ..." if len(nobody) > 10 else ""), file=out)
    print(f"\noutput: {output}", file=out)


if __name__ == "__main__":
    main()