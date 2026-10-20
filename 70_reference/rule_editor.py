#!/usr/bin/env python3
"""Editor of verification rules for confirmed harmful spans, with an LLM verifier.

For every human-confirmed harmful span (one editor item), the page shows
  - the source (gold transcript) and the ASR the reference system translated,
  - all target translations: the reference one (e.g. canary_cs) with the span highlighted,
    and the other systems (other_outputs),
  - an editable box with one or more verification rules (one rule per line).
  - editable human contrastive translations (add/remove any number), each with an editable label
    (default 'contrastive 1', 'contrastive 2', ...), below the systems and verified with the
    same rules. Saved as "contrastive": [{"label", "text"}]; their verdicts are under the label.
[Verify] sends every (target, rule) pair to the LLM verifier, which answers pass/fail.
A target is harmful if it fails at least one rule. The reference target is expected to fail;
the others are compared with the human same_harm labels.

Rules and verdicts are saved to --output (JSONL, one line per span; rewritten on every save,
and loaded again on start, so you can stop and continue).

Usage:
  python rule_editor.py data.jsonl --output rules_verified.jsonl \
      --backend einfra --model gpt-oss-120B
  # then open http://localhost:8765  (on a cluster: ssh -L 8765:localhost:8765 host)

  --rules verification_rules.jsonl   prefill rules ({"rule_id": "<doc>.<seg>.rN",
                                     "canary_span": [...], "rule": "..."} per line)
  --backend dummy                    no model: the reference target fails, the others pass
  --cache FILE / --no-cache          LLM answers are cached on disk (default <output>.llm-cache.jsonl),
                                     so Verify only asks the LLM about new or changed (target, rule)
                                     pairs; the page has a checkbox to ask again anyway
  --audio-root DIR                   where the relative 'audio' paths start (default: the current
                                     directory, then the directory of the data file)
The backend options (--backend, --model, --api-base, ...) come from llm_backends.py,
which must be importable (same directory or PYTHONPATH); see --backend-module.
"""
import argparse
import importlib
import contextlib
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_PROMPT = """You check one translation against one verification rule.

SOURCE (English): {source}
TRANSLATION (Czech): {translation}
VERIFICATION RULE: {rule}

Judge only what the rule is about. Ignore all other errors in the translation.
Answer with JSON: {{"evidence": "<the part of the translation the rule applies to, quoted, or 'omitted'>", "verdict": "pass" | "fail"}}"""

SCHEMA = {
    "type": "object",
    "properties": {"evidence": {"type": "string"},
                   "verdict": {"type": "string", "enum": ["pass", "fail"]}},
    "required": ["evidence", "verdict"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------- data

def seg_id(d):
    """'431464443.0045' from document '431464443_2397.80_2996.52' and audio '....0045.wav'."""
    m = re.search(r"\.(\d+)\.wav$", d.get("audio", ""))
    return d["document"].split("_")[0] + "." + (m.group(1) if m else "?")


def locate(text, span):
    """(start, end) of the span in text, trusting span_start/span_end if they match."""
    s, e = span.get("span_start"), span.get("span_end")
    if isinstance(s, int) and isinstance(e, int) and text[s:e] == span["span"]:
        return s, e
    i = text.find(span["span"])
    return (i, i + len(span["span"])) if i >= 0 else (None, None)


def load_items(path):
    """One item per (segment, reference target, confirmed span)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            sid = seg_id(d)
            for t in d.get("targets", []):
                spans = t.get("harm_spans", [])
                marks = [locate(t["text"], s) for s in spans]
                for i, sp in enumerate(spans):
                    items.append({
                        "key": f"{sid}|{t['system']}|{i}",
                        "seg": sid, "line": n, "document": d["document"], "audio": d.get("audio"),
                        "source": d.get("gold_transcript", ""),
                        "asr": d.get("asr", ""), "asr_system": d.get("asr_system", ""),
                        "ref_system": t["system"], "ref_text": t["text"],
                        "span_index": i, "span": sp["span"],
                        "span_start": marks[i][0], "span_end": marks[i][1],
                        "other_spans": [m for j, m in enumerate(marks) if j != i and m[0] is not None],
                        "same_harm": sp.get("same_harm", {}),
                        "ai": [a for a in t.get("ai_suggested_spans", [])],
                        "others": d.get("other_outputs", {}),
                    })
    return items


def load_prefill(path, items):
    """Rules from a verification_rules.jsonl (rule_id '<seg>.rN', canary_span, rule)."""
    by_seg = {}
    for it in items:
        by_seg.setdefault(it["seg"], []).append(it)
    rules = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = re.sub(r"\.r\d+$", "", r.get("rule_id", ""))
            spans = r.get("canary_span") or []
            spans = [spans] if isinstance(spans, str) else spans
            for it in by_seg.get(sid, []):
                if it["span"] in spans:
                    rules.setdefault(it["key"], []).append(r["rule"])
    return rules


# ---------------------------------------------------------------------------- verifier

def parse_verdict(text):
    """{'verdict': 'pass'|'fail'|'error', 'evidence': ...} from a model answer."""
    if text is None:
        return {"verdict": "error", "evidence": "request failed"}
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", t, flags=re.S))):
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        v = str(o.get("verdict", "")).strip().lower()
        if v in ("pass", "fail"):
            return {"verdict": v, "evidence": str(o.get("evidence", ""))}
    m = re.search(r"\b(pass|fail)\b", t, flags=re.I)
    if m:
        return {"verdict": m.group(1).lower(), "evidence": "", "raw": text}
    return {"verdict": "error", "evidence": "unparsable answer", "raw": text}


class VerifyError(Exception):
    """The LLM could not be used; the message is shown in the editor."""


class Tee(io.TextIOBase):
    """Write to the real stderr and keep a copy (the backend reports failures on stderr)."""

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


class Cache:
    """Disk cache of LLM answers: one JSON line per answer, appended as they come.
    The key is a hash of everything that determines the answer: backend, model, decoding settings
    and the full chat (prompt with source, translation and rule filled in). Only answers with a
    pass/fail verdict are stored, so failed or unparsable requests are asked again next time."""

    def __init__(self, path):
        self.path, self.data, self.lock = path, {}, threading.Lock()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.data[r["key"]] = r["answer"]
                    except (json.JSONDecodeError, KeyError):
                        pass  # a line cut off by a crash
            print(f"cache: {len(self.data)} LLM answers in {path}", file=sys.stderr)

    @staticmethod
    def key(settings, chat):
        return hashlib.sha256(json.dumps([settings, chat], ensure_ascii=False, sort_keys=True)
                              .encode("utf-8")).hexdigest()

    def get(self, key):
        return self.data.get(key)

    def put_many(self, entries, model):
        """entries: [(key, answer)]"""
        if not self.path or not entries:
            return
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            for k, a in entries:
                self.data[k] = a
                f.write(json.dumps({"key": k, "model": model, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    "answer": a}, ensure_ascii=False) + "\n")


class Verifier:
    def __init__(self, args, mod):
        self.prompt = DEFAULT_PROMPT
        if args.prompt:
            with open(args.prompt, encoding="utf-8") as f:
                self.prompt = f.read()
        self.mod = mod
        self.cfg = mod.BackendConfig.from_args(args)
        self.dummy = self.cfg.backend == "dummy"
        self.backend, self.init_error = None, None
        self.lock = threading.Lock()   # vLLM is not thread-safe; one verification at a time
        self.model = "dummy" if self.dummy else f"{self.cfg.backend}:{self.cfg.model}"
        # what changes the answer, besides the chat itself
        self.settings = {"backend": self.cfg.backend, "model": self.cfg.model,
                         "temperature": self.cfg.temperature, "max_new_tokens": self.cfg.max_new_tokens,
                         "guided_json": self.cfg.guided_json}
        self.cache = Cache(None if (self.dummy or args.no_cache) else
                           (args.cache or os.path.splitext(args.output)[0] + ".llm-cache.jsonl"))
        if not self.dummy:
            self._init_backend()

    def _init_backend(self):
        """Load the backend; on failure keep the editor running and remember the error
        (shown in the page; the next Verify tries again)."""
        tee = Tee(sys.stderr)
        try:
            with contextlib.redirect_stderr(tee):
                self.backend = self.mod.get_backend(self.cfg)
            self.init_error = None
        except (Exception, SystemExit) as e:
            self.backend = None
            log = tee.text()
            self.init_error = (f"Could not start the LLM backend {self.model}: {describe(e)}"
                               + (f"\n\n{log}" if log else ""))
            print(self.init_error, file=sys.stderr)

    def chat(self, source, translation, rule):
        content = (self.prompt.replace("{source}", source).replace("{translation}", translation)
                   .replace("{rule}", rule).replace("{{", "{").replace("}}", "}"))
        return [{"role": "user", "content": content}]

    def verify(self, item, rules, contrastive=(), use_cache=True):
        """({system: [verdict per rule]}, [error messages], {"cached": n, "asked": m}) for the
        reference, all other systems and the human contrastive translations (under their labels).
        With use_cache, pairs answered before come from the disk cache; new answers are added to it."""
        targets = {item["ref_system"]: item["ref_text"], **item["others"],
                   **{c["label"]: c["text"] for c in contrastive if c["text"].strip()}}
        jobs = [(s, k) for s in targets for k in range(len(rules))]
        reasons = []
        stats = {"cached": 0, "asked": 0}
        if self.dummy:
            answers = [json.dumps({"evidence": "(dummy)",
                                   "verdict": "fail" if s == item["ref_system"] else "pass"})
                       for s, _ in jobs]
            stats["asked"] = len(jobs)
        else:
            chats = [self.chat(item["source"], targets[s], rules[k]) for s, k in jobs]
            keys = [Cache.key(self.settings, c) for c in chats]
            answers = [self.cache.get(k) if use_cache else None for k in keys]
            todo = [i for i, a in enumerate(answers) if a is None]
            stats = {"cached": len(jobs) - len(todo), "asked": len(todo)}
            if todo:
                with self.lock:
                    if self.backend is None:
                        self._init_backend()
                        if self.backend is None:
                            raise VerifyError(self.init_error)
                    tee = Tee(sys.stderr)
                    try:
                        with contextlib.redirect_stderr(tee):
                            new = list(self.backend.generate([chats[i] for i in todo], SCHEMA))
                    except (Exception, SystemExit) as e:
                        log = tee.text()
                        raise VerifyError(f"The LLM call failed: {describe(e)}" + (f"\n\n{log}" if log else ""))
                log = tee.text()
                # the API backends return None for a request that failed and print why on stderr
                # (parallel requests can interleave their lines, hence the regex)
                reasons = [r.strip() for r in re.findall(r"WARNING: request failed: (.*?)(?=WARNING:|\n|$)", log)]
                failed = sum(a is None for a in new)
                if failed == len(jobs):
                    raise VerifyError(f"All {failed} LLM requests failed; nothing was saved.\n\n"
                                      + (summarize_reasons(reasons) or log or "no details from the backend"))
                for i, a in zip(todo, new):
                    answers[i] = a
                self.cache.put_many([(keys[i], a) for i, a in zip(todo, new)
                                     if a is not None and parse_verdict(a)["verdict"] != "error"], self.model)
        res = {s: [None] * len(rules) for s in targets}
        errors = []
        for (s, k), a in zip(jobs, answers):
            v = parse_verdict(a)
            if a is None and not self.dummy:
                v["evidence"] = reasons[0] if reasons else "request failed"
            res[s][k] = v
        n_failed = sum(v["verdict"] == "error" and v["evidence"] != "unparsable answer"
                       for vs in res.values() for v in vs)
        n_unparsed = sum(v["verdict"] == "error" and v["evidence"] == "unparsable answer"
                         for vs in res.values() for v in vs)
        if n_failed:
            errors.append(f"{n_failed} of {len(jobs)} LLM requests failed (marked ?)."
                          + (f"\n{summarize_reasons(reasons)}" if reasons else ""))
        if n_unparsed:
            raw = next(v.get("raw") for vs in res.values() for v in vs if v["evidence"] == "unparsable answer")
            errors.append(f"{n_unparsed} LLM answer(s) had no pass/fail verdict (marked ?). "
                          f"First one:\n{(raw or '')[:1500]}")
        return res, errors, stats


def normalize_contrastive(raw, item):
    """[{"label", "text"}] with non-empty labels that are unique and differ from the system names.
    Accepts plain strings too (older saved files)."""
    taken = {item["ref_system"], *item["others"]}
    out = []
    for j, c in enumerate(raw or []):
        if isinstance(c, dict):
            label, text = str(c.get("label", "")).strip(), str(c.get("text", ""))
        else:
            label, text = "", str(c)
        label = label or f"contrastive {j + 1}"
        base, n = label, 2
        while label in taken:
            label, n = f"{base} ({n})", n + 1
        taken.add(label)
        out.append({"label": label, "text": text})
    return out


# ---------------------------------------------------------------------------- store

class Store:
    def __init__(self, path, items, prefill):
        self.path, self.lock = path, threading.Lock()
        self.records = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.records[r["key"]] = r
            print(f"loaded {len(self.records)} saved items from {path}", file=sys.stderr)
        self.order = [it["key"] for it in items]
        self.prefill = prefill

    def rules(self, key):
        r = self.records.get(key)
        return r["rules"] if r else self.prefill.get(key, [])

    def put(self, item, rules, contrastive, results=None, model=None):
        old = self.records.get(item["key"])
        rec = {
            "key": item["key"], "document": item["document"], "audio": item["audio"],
            "seg": item["seg"], "ref_system": item["ref_system"], "span_index": item["span_index"],
            "span": item["span"], "span_start": item["span_start"], "span_end": item["span_end"],
            "source": item["source"], "rules": rules, "contrastive": contrastive,
            "human_same_harm": item["same_harm"],
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if results is not None:
            rec.update(verified_with=model, results=results, **summarize(item, rules, results))
        elif old and old.get("rules") == rules and "results" in old:
            # rules unchanged: keep the verdicts of the systems, and of the contrastive translations
            # whose text did not change (found by text, so renaming a label keeps its verdicts)
            systems = {item["ref_system"], *item["others"]}
            res = {s: v for s, v in old["results"].items() if s in systems}
            by_text = {c["text"]: old["results"].get(c["label"])
                       for c in normalize_contrastive(old.get("contrastive"), item)}
            for c in contrastive:
                if by_text.get(c["text"]):
                    res[c["label"]] = by_text[c["text"]]
            rec.update(verified_with=old.get("verified_with"), results=res, **summarize(item, rules, res))
        with self.lock:
            self.records[item["key"]] = rec
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for k in self.order + [k for k in self.records if k not in self.order]:
                    if k in self.records:
                        f.write(json.dumps(self.records[k], ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
        return rec


def summarize(item, rules, results):
    harmful = {s: any(v["verdict"] == "fail" for v in vs) for s, vs in results.items()}
    ref = results[item["ref_system"]]
    agree = {s: harmful[s] == bool(h) for s, h in item["same_harm"].items() if s in harmful}
    return {"harmful": harmful,
            "ref_fails_each_rule": [v["verdict"] == "fail" for v in ref],
            "agreement": agree}


def status(rec):
    """Sidebar mark: '' no rules, 'edit' not verified, 'bad' ref passes a rule or disagrees, 'ok'."""
    if not rec or not rec.get("rules"):
        return ""
    if "results" not in rec:
        return "edit"
    if not all(rec["ref_fails_each_rule"]) or not all(rec["agreement"].values()):
        return "bad"
    return "ok"


# ---------------------------------------------------------------------------- web

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Verification rules</title>
<style>
:root{--bg:#fbfbfa;--panel:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e3e2de;--hl:#ffd9a8;--hl2:#f3ecdf;
--pass:#1f7a4a;--passbg:#e3f3ea;--fail:#b3261e;--failbg:#fbe5e3;--err:#7a5b00;--errbg:#fff3c4;--acc:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--panel:#212124;--ink:#ececea;--mute:#9a9a95;--line:#36363a;
--hl:#7a4a12;--hl2:#3a3328;--pass:#6fd39c;--passbg:#1c3527;--fail:#ff8a80;--failbg:#40201e;--err:#f0d06a;--errbg:#3b3315;--acc:#86a8ff}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--ink);display:flex;height:100vh}
#side{width:260px;border-right:1px solid var(--line);overflow:auto;background:var(--panel);flex:none}
#side input{width:calc(100% - 16px);margin:8px;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.it{padding:5px 10px;cursor:pointer;border-left:3px solid transparent;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.it:hover{background:var(--hl2)}.it.cur{border-left-color:var(--acc);background:var(--hl2)}
.it .m{display:inline-block;width:14px;font-weight:700}.m.ok{color:var(--pass)}.m.bad{color:var(--fail)}.m.edit{color:var(--err)}
#main{flex:1;overflow:auto;padding:18px 26px 60px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:18px 0 6px}
.box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tt{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:8px}
.tt th,.tt td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
.tt tr:last-child td{border-bottom:0}.tt th{font-size:12px;color:var(--mute);font-weight:600}
.tt td.sys{font-weight:600;font-size:12.5px;white-space:nowrap;width:1%}
.tt th.r,.tt td.r{text-align:center;width:38px;font-size:16px;font-weight:700;cursor:default}
.tt input.cl{font:inherit;font-weight:600;width:150px;padding:3px 6px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:var(--ink)}
.tt textarea.ct{min-height:0;font:inherit;padding:5px 7px;resize:none;overflow:hidden}
button.x{font-size:11.5px;padding:1px 7px;margin-top:4px;color:var(--mute)}
.ok{color:var(--pass)}.no{color:var(--fail)}.er{color:var(--err)}
audio{width:100%;max-width:520px;height:34px;margin-top:10px}
mark{background:var(--hl);color:inherit;padding:0 2px;border-radius:3px}mark.o{background:none;border-bottom:2px dashed var(--mute)}
textarea{width:100%;min-height:130px;font:13.5px/1.5 ui-monospace,monospace;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink);resize:vertical}
button{font:inherit;padding:7px 14px;border-radius:7px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button.p{background:var(--acc);color:#fff;border-color:var(--acc)}button:disabled{opacity:.5;cursor:wait}
.bar{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}.msg{color:var(--mute);font-size:12.5px}
table{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;font-size:13px}
th{font-size:12px;color:var(--mute);font-weight:600}
.v{display:inline-block;padding:1px 7px;border-radius:10px;font-weight:600;font-size:12px}
.v.pass{color:var(--pass);background:var(--passbg)}.v.fail{color:var(--fail);background:var(--failbg)}.v.error{color:var(--err);background:var(--errbg)}
.ev{color:var(--mute);font-size:12px;margin-top:2px}.warn{color:var(--fail);font-weight:600}
.ai{font-size:12.5px;color:var(--mute);margin-top:4px}.ai b{color:var(--ink);font-weight:600}
.nav{display:flex;gap:8px;align-items:center}.head{display:flex;justify-content:space-between;align-items:center;gap:10px}
.legend{font-size:12px;color:var(--mute)}
.errbox{margin:10px 0 0;padding:10px 12px;border:1px solid var(--fail);border-radius:8px;background:var(--failbg);color:var(--fail);
 font:12.5px/1.45 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto}
</style></head><body>
<div id="side"><input id="flt" placeholder="filter (id, span, mark: ! ✓ ✎)"><div id="list"></div></div>
<div id="main">
 <div class="head"><div><b id="title"></b> <span class="msg" id="pos"></span></div>
  <div class="nav"><button onclick="go(cur-1)">← prev</button><button onclick="go(cur+1)">next →</button></div></div>
 <audio id="aud" controls preload="none"></audio>
 <h2>Source (gold transcript)</h2><div class="box" id="src"></div>
 <h2>ASR given to <span id="refsys2"></span></h2><div class="box msg" id="asr"></div>
 <h2>Targets and verdicts</h2><div id="tgts"></div>
 <h2>Verification rules <span class="legend">— one rule per line (R1, R2, …); Ctrl+Enter = verify, Ctrl+S = save</span></h2>
 <textarea id="rules" spellcheck="false"></textarea>
 <div class="bar"><button class="p" id="vbtn" onclick="verify()">Verify</button>
  <button onclick="save()">Save</button>
  <label class="msg" title="ask the LLM again for every pair, and refresh the cache"><input type="checkbox" id="nocache"> ignore cache</label>
  <span class="msg" id="msg"></span></div>
 <pre class="errbox" id="err" hidden></pre>
 <div class="msg" id="vnote"></div>
</div>
<script>
let items=[],cur=0,item=null,dirty=false,contr=[];  // contr: the human contrastive translations being edited
const $=id=>document.getElementById(id), esc=s=>(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const MARK={'':'','edit':'✎','bad':'!','ok':'✓'};
async function api(p,body){let r;
 try{r=await fetch(p,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})}
 catch(e){throw new Error('The editor server is not reachable ('+e.message+'). Is rule_editor.py still running?')}
 const t=await r.text();let j=null;try{j=JSON.parse(t)}catch(e){}
 if(!r.ok)throw new Error(j&&j.error?j.error:`HTTP ${r.status}: ${t}`);return j}
function showErr(m){const b=$('err');b.textContent=m||'';b.hidden=!m}
function renderList(){const f=$('flt').value.toLowerCase();
 $('list').innerHTML=items.map((it,i)=>{const t=`${MARK[it.status]} ${it.seg} ${it.span}`.toLowerCase();
  if(f&&!t.includes(f))return'';
  return `<div class="it${i==cur?' cur':''}" onclick="go(${i})" title="${esc(it.span)}"><span class="m ${it.status}">${MARK[it.status]}</span>${it.seg} · ${esc(it.span)}</div>`}).join('')}
function hl(text,s,e,others){ // highlight current span [s,e) and other confirmed spans
 const marks=[];if(s!=null)marks.push([s,e,'']);(others||[]).forEach(o=>marks.push([o[0],o[1],'o']));
 marks.sort((a,b)=>a[0]-b[0]);let out='',p=0;
 for(const [a,b,c] of marks){if(a<p)continue;out+=esc(text.slice(p,a))+`<mark class="${c}">${esc(text.slice(a,b))||'∅'}</mark>`;p=b}
 return out+esc(text.slice(p))}
async function go(i){if(i<0||i>=items.length)return;
 if(dirty&&!confirm('Rules changed and not saved. Leave?'))return;
 cur=i;item=await api('/api/item?i='+i);dirty=false;location.hash=i;
 $('title').textContent=item.seg+' · '+item.ref_system+' · span '+(item.span_index+1);$('pos').textContent=`${i+1} / ${items.length}`;
 $('src').textContent=item.source;$('asr').textContent=item.asr;$('refsys2').textContent=item.ref_system+' ('+(item.asr_system||'asr')+')';
 const a=$('aud');a.pause();if(item.has_audio){a.src='/api/audio?i='+i;a.style.display=''}else{a.removeAttribute('src');a.style.display='none'}
 contr=((item.record&&item.record.contrastive)||[]).map(c=>({...c}));
 $('rules').value=(item.rules||[]).join('\n');$('msg').textContent='';showErr(item.backend_error);renderRes(item.record);renderList();
 document.querySelector('.it.cur')?.scrollIntoView({block:'nearest'})}
function savedFor(rec,s,c){ // saved verdicts of system s, or of contrastive c (matched by its text), if still valid
 if(!rec||!rec.results||JSON.stringify(rec.rules)!==JSON.stringify(rulesNow()))return null;
 if(c){const o=(rec.contrastive||[]).find(o=>o.text===c.text);return o?rec.results[o.label]||null:null}
 return rec.results[s]||null}
function freshFor(rec,s){return !!savedFor(rec,s,null)}
function ticks(rec,s,c,n){let h='';const vs=savedFor(rec,s,c);
 for(let k=0;k<n;k++){const v=vs?vs[k]:null;
  if(!v){h+='<td class="r msg">·</td>';continue}
  const [c,g]=v.verdict=='pass'?['ok','✓']:v.verdict=='fail'?['no','✗']:['er','?'];
  h+=`<td class="r ${c}" title="${esc('R'+(k+1)+': '+rec.rules[k]+'\n\n'+v.verdict.toUpperCase()+': '+(v.evidence||'')+(v.raw?'\n\nraw: '+v.raw:''))}">${g}</td>`}
 return h}
function nRules(){return rulesNow().length||(item.record&&item.record.rules?item.record.rules.length:0)}
function renderRes(rec){ // table: system | target | one tick per rule; human contrastive rows at the bottom
 const sys=[item.ref_system,...Object.keys(item.others)],n=nRules();
 let t='<table class="tt"><tr><th>system</th><th>target</th>'+Array.from({length:n},(_,k)=>`<th class="r">R${k+1}</th>`).join('')+'</tr>';
 for(const s of sys)t+=`<tr><td class="sys">${esc(s)}</td><td>${s==item.ref_system?hl(item.ref_text,item.span_start,item.span_end,item.other_spans):esc(item.others[s])}</td>${ticks(rec,s,null,n)}</tr>`;
 contr.forEach((c,j)=>{t+=`<tr><td class="sys"><input class="cl" data-j="${j}" value="${esc(c.label)}" title="label (editable)"><br><button class="x" onclick="delContr(${j})" title="remove this contrastive translation">remove</button></td>`
   +`<td><textarea class="ct" data-j="${j}" rows="2" placeholder="human contrastive translation">${esc(c.text)}</textarea></td>${ticks(rec,null,c,n)}</tr>`});
 t+=`<tr><td colspan="${2+n}"><button onclick="addContr()">+ add contrastive translation</button></td></tr>`;
 $('tgts').innerHTML=t+'</table>';
 document.querySelectorAll('.ct').forEach(grow);
 let note='';
 if(rec&&rec.results&&rec.ref_fails_each_rule){const miss=rec.rules.map((r,k)=>rec.ref_fails_each_rule[k]?null:'R'+(k+1)).filter(Boolean);
  note=(miss.length?`<span class="warn">${esc(item.ref_system)} passes ${miss.join(', ')}: the rule does not catch the confirmed error.</span> `:'')
   +`Verified with ${esc(rec.verified_with)} at ${esc(rec.saved_at)}.`+(freshFor(rec,item.ref_system,null)?'':' <span class="warn">Rules edited since; verify again.</span>')}
 $('vnote').innerHTML=note}
function grow(el){el.style.height='auto';el.style.height=(el.scrollHeight+2)+'px'}
function freeLabel(){const used=new Set([item.ref_system,...Object.keys(item.others),...contr.map(c=>c.label)]);
 let k=1;while(used.has('contrastive '+k))k++;return 'contrastive '+k}
function addContr(){contr.push({label:freeLabel(),text:''});dirty=true;renderRes(item.record);const a=document.querySelectorAll('.ct');a[a.length-1].focus()}
function delContr(j){if(contr[j].text.trim()&&!confirm('Remove '+contr[j].label+'?'))return;contr.splice(j,1);dirty=true;renderRes(item.record)}
document.addEventListener('input',e=>{const el=e.target,j=+el.dataset.j;
 if(el.classList.contains('cl')){contr[j].label=el.value;dirty=true;return}   // renaming keeps the verdicts
 if(!el.classList.contains('ct'))return;
 contr[j].text=el.value;dirty=true;grow(el);
 el.closest('tr').querySelectorAll('td.r').forEach(td=>{td.className='r msg';td.textContent='·';td.removeAttribute('title')})});
const rulesNow=()=>$('rules').value.split('\n').map(s=>s.trim()).filter(Boolean);
async function save(){try{const r=await api('/api/save',{i:cur,rules:rulesNow(),contrastive:contr});dirty=false;contr=r.record.contrastive.map(c=>({...c}));items[cur].status=r.status;item.record=r.record;renderRes(r.record);renderList();$('msg').textContent='saved'}catch(e){$('msg').textContent='not saved';showErr(e.message)}}
async function verify(){const rules=rulesNow();if(!rules.length){$('msg').textContent='write at least one rule';return}
 $('vbtn').disabled=true;showErr('');const t0=Date.now();$('msg').textContent='verifying '+rules.length+' rule(s) × '+(1+Object.keys(item.others).length+contr.filter(c=>c.text.trim()).length)+' targets…';
 try{const r=await api('/api/verify',{i:cur,rules,contrastive:contr,no_cache:$('nocache').checked});dirty=false;contr=r.record.contrastive.map(c=>({...c}));items[cur].status=r.status;item.record=r.record;renderRes(r.record);renderList();
  const st=r.stats||{};$('msg').textContent=`verified and saved (${((Date.now()-t0)/1000).toFixed(1)} s; ${st.cached||0} from cache, ${st.asked||0} asked)`+(r.errors&&r.errors.length?', with errors':'');
  showErr((r.errors||[]).join('\n\n'))}catch(e){$('msg').textContent='verification failed';showErr(e.message)}
 $('vbtn').disabled=false}
$('rules').addEventListener('input',()=>{dirty=true;if(item)renderRes(item.record)});$('flt').addEventListener('input',renderList);
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key=='Enter'){e.preventDefault();verify()}
 if((e.ctrlKey||e.metaKey)&&e.key=='s'){e.preventDefault();save()}
 if(e.altKey&&e.key=='ArrowDown'){e.preventDefault();go(cur+1)}if(e.altKey&&e.key=='ArrowUp'){e.preventDefault();go(cur-1)}});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue=''}});
(async()=>{items=await api('/api/items');go(Math.min(parseInt(location.hash.slice(1))||0,items.length-1))})();
</script></body></html>"""


def make_handler(items, store, verifier, audio_roots):
    def audio_path(it):
        """The item's audio file: as given, or relative to one of the roots."""
        a = it.get("audio")
        if not a:
            return None
        for root in [""] + audio_roots:
            p = os.path.normpath(os.path.join(root, a))
            if os.path.isfile(p):
                return p
        return None

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False))

        def _item(self, i):
            if not isinstance(i, int) or not 0 <= i < len(items):
                raise ValueError(f"no item {i}")
            return items[i]

        def _audio(self, path):
            """Serve an audio file, with HTTP Range so that the player can seek."""
            if not path:
                return self._send(404, "audio not found", "text/plain")
            size = os.path.getsize(path)
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else end
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(end - start + 1)
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(data)))
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/":
                    self._send(200, PAGE, "text/html")
                elif u.path == "/api/items":
                    self._json([{"seg": it["seg"], "span": it["span"],
                                 "status": status(store.records.get(it["key"])) or
                                 ("edit" if store.prefill.get(it["key"]) else "")}
                                for it in items])
                elif u.path == "/api/audio":
                    self._audio(audio_path(self._item(int(parse_qs(u.query)["i"][0]))))
                elif u.path == "/api/item":
                    it = self._item(int(parse_qs(u.query)["i"][0]))
                    rec = store.records.get(it["key"])
                    if rec:  # older files: contrastive as plain strings
                        rec["contrastive"] = normalize_contrastive(rec.get("contrastive"), it)
                    self._json({**it, "rules": store.rules(it["key"]), "backend_error": verifier.init_error, "has_audio": bool(audio_path(it)),
                                "record": store.records.get(it["key"])})
                else:
                    self._send(404, "not found", "text/plain")
            except Exception as e:
                self._send(400, str(e), "text/plain")

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                it = self._item(body["i"])
                rules = [str(r).strip() for r in body.get("rules", []) if str(r).strip()]
                contrastive = normalize_contrastive(body.get("contrastive"), it)
                if self.path == "/api/save":
                    rec = store.put(it, rules, contrastive)
                elif self.path == "/api/verify":
                    t0 = time.time()
                    try:
                        results, errors, stats = verifier.verify(it, rules, contrastive,
                                                                 use_cache=not body.get("no_cache"))
                    except VerifyError as e:
                        return self._json({"error": str(e)}, 502)
                    rec = store.put(it, rules, contrastive, results, verifier.model)
                    print(f"{it['key']}: {len(rules)} rule(s) verified in {time.time() - t0:.1f}s "
                          f"({stats['cached']} cached, {stats['asked']} asked)", file=sys.stderr)
                    return self._json({"record": rec, "status": status(rec), "errors": errors, "stats": stats})
                else:
                    return self._send(404, "not found", "text/plain")
                self._json({"record": rec, "status": status(rec)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send(500, f"{type(e).__name__}: {e}", "text/plain")

    return H


def main():
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--backend-module", default="llm_backends")
    known, _ = pre.parse_known_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    mod = importlib.import_module(known.backend_module)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="annotated JSONL (targets with harm_spans, other_outputs)")
    p.add_argument("--output", "-o", required=True, help="JSONL with rules and verdicts (loaded if it exists)")
    p.add_argument("--rules", help="prefill rules from a verification_rules.jsonl")
    p.add_argument("--prompt", help="verifier prompt file with {source}, {translation}, {rule}")
    p.add_argument("--audio-root", action="append", default=[],
                   help="directory the 'audio' paths are relative to (repeatable; tried after the "
                        "current directory and the directory of the data file)")
    p.add_argument("--cache", help="disk cache of LLM answers (default: <output>.llm-cache.jsonl)")
    p.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--backend-module", default="llm_backends",
                   help="module with BackendConfig, add_arguments, get_backend (default: llm_backends)")
    mod.add_arguments(p)
    args = p.parse_args()

    items = load_items(args.data)
    if not items:
        sys.exit(f"no confirmed harm_spans in {args.data}")
    prefill = load_prefill(args.rules, items) if args.rules else {}
    store = Store(args.output, items, prefill)
    print(f"{len(items)} spans; {len(prefill)} prefilled from --rules", file=sys.stderr)
    roots = args.audio_root + [os.path.dirname(os.path.abspath(args.data))]
    found = sum(any(os.path.isfile(os.path.join(r, it["audio"])) for r in [""] + roots)
                for it in items if it.get("audio"))
    print(f"audio found for {found}/{len(items)} spans" + ("" if found else
          "; set --audio-root to the directory the 'audio' paths are relative to"), file=sys.stderr)
    verifier = Verifier(args, mod)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(items, store, verifier,
                                                         args.audio_root + [os.path.dirname(os.path.abspath(args.data))]))
    print(f"open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()