#!/usr/bin/env python3
"""Editor of verification rules for confirmed harmful spans, with an LLM verifier.

For every human-confirmed harmful span (one editor item), the page shows
  - the audio and the source (gold transcript),
  - all target translations: the reference one (e.g. canary_cs) with the span highlighted,
    and the other systems (from --systems: every text in the target language; else other_outputs),
  - an editable box with one or more verification rules (one rule per line).
  - editable human contrastive translations (add/remove any number), each with an editable label
    (default 'contrastive 1', 'contrastive 2', ...), below the systems and verified with the
    same rules. Saved as "contrastive": [{"label", "text"}]; their verdicts are under the label.
[Verify] sends every (target, rule) pair to the LLM verifier, which answers pass/fail.
Each span has a status: pending (not touched yet), in progress (saved or verified), finished
(the annotator ticked remove, undecidable or finished). Finished needs at least one rule, current
verdicts, and for every rule at least one translation (a system or a contrastive one) that passes it.
A target is harmful if it fails at least one rule. The reference target is expected to fail;
the others are compared with the human same_harm labels.

At startup, all rules (saved in --output, or prefilled from --rules) are verified against all
targets with the LLM, answers in the cache are reused (--no-startup-verify skips this).
Edits are autosaved 1.5 s after the last keystroke (and when leaving the item or closing the tab).
Rules and verdicts are saved to --output (JSONL, one line per span; rewritten on every save,
and loaded again on start, so you can stop and continue).

Usage:
  python rule_editor.py data.jsonl --output rules_verified.jsonl \
      --backend einfra --model gpt-oss-120B
  # then open http://localhost:8765  (on a cluster: ssh -L 8765:localhost:8765 host)

  --systems systems.jsonl            translations of the other systems ({"document", "audio",
                                     "systems_info": {sys: {"lan"}}, "text": {sys: ...}} per segment)
  --rules verification_rules.jsonl   prefill rules ({"rule_id": "<doc>.<seg>.rN",
                                     "canary_span": [...], "rule": "..."} per line)
  --backend dummy                    no model: the reference target fails, the others pass
  --cache FILE / --no-cache          LLM answers are cached on disk (default <output>.llm-cache.jsonl),
                                     so Verify only asks the LLM about new or changed (target, rule)
                                     pairs; the page has a checkbox to ask again anyway
  --audio-root DIR                   where the relative 'audio' paths start (default: the current
                                     directory, then the directory of the data file)
The LLM verifier (prompt, answer parsing, cache) is in llm_verifier.py, which also runs the rules
in batch. The backend options (--backend, --model, --api-base, ...) come from llm_backend.py.
Both must be importable (same directory or PYTHONPATH).
"""
import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import llm_backend
from llm_verifier import (ProgressLog, Verifier, VerifyError, add_thinking_argument, load_systems,
                          parse_verdict, read_prompt, seg_key, summarize_reasons, DEFAULT,
                          THINKING_LEVELS,
                          target_texts)

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


def load_items(path, systems=None):
    """One item per (segment, reference target, confirmed span).
    systems: from load_systems(); the other systems come from there instead of other_outputs."""
    items = []
    missing, used = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            sid = seg_id(d)
            sysline = None
            if systems is not None:
                sysline = systems.get(seg_key(d))
                if sysline is None:
                    missing.append(f"{sid} (line {n}, {d.get('audio')})")
                else:
                    used.add(seg_key(d))
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
                        "others": ({s: x for s, x in target_texts(sysline, t["system"], t.get("tgt_lan"),
                                                               skip=("gold_transcript", d.get("asr_system"))).items()
                                    if s != t["system"]} if sysline else d.get("other_outputs", {})),
                    })
    if systems is not None:
        for m in missing[:20]:
            print(f"WARNING: confirmed segment not in --systems, shown without other systems: {m}",
                  file=sys.stderr)
        if len(missing) > 20:
            print(f"WARNING: ... and {len(missing) - 20} more such segments", file=sys.stderr)
        print(f"--systems: {len(used)} segments used, {len(systems) - len(used)} skipped (no confirmed "
              f"spans), {len(missing)} confirmed segments missing", file=sys.stderr)
    return items


def load_prefill(path, items):
    """Rules from a verification_rules.jsonl, one rule per line: matched by "key" (the item key,
    as written by synthesize_rules.py) if present, else by rule_id '<seg>.rN' and canary_span."""
    by_seg = {}
    for it in items:
        by_seg.setdefault(it["seg"], []).append(it)
    keys = {it["key"] for it in items}
    rules = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("key") in keys:
                rules.setdefault(r["key"], []).append(r["rule"])
                continue
            sid = re.sub(r"\.r\d+$", "", r.get("rule_id", ""))
            spans = r.get("canary_span") or []
            spans = [spans] if isinstance(spans, str) else spans
            for it in by_seg.get(sid, []):
                if it["span"] in spans:
                    rules.setdefault(it["key"], []).append(r["rule"])
    return rules


# ---------------------------------------------------------------------------- verifier

def verify_item(verifier, item, rules, contrastive=(), use_cache=True, thinking=DEFAULT):
    """({target: [verdict per rule]}, [error messages], stats) for the reference, the other systems
    and the non-empty contrastive translations (under their labels) of an editor item.
    thinking: the level for these requests (DEFAULT: --enable-thinking)."""
    targets = {item["ref_system"]: item["ref_text"], **item["others"],
               **{c["label"]: c["text"] for c in contrastive if c["text"].strip()}}
    return verifier.verify(item["source"], targets, rules, use_cache, dummy_fail={item["ref_system"]},
                           thinking=thinking)


def normalize_contrastive(raw, item):
    """[{"label", "text", "confirmed"}] with non-empty labels that are unique and differ from the system
    names; confirmed: the annotator's pass/fail for it (None = not chosen yet).
    Accepts plain strings too (older saved files)."""
    taken = {item["ref_system"], *item["others"]}
    out = []
    for j, c in enumerate(raw or []):
        conf = None
        if isinstance(c, dict):
            label, text = str(c.get("label", "")).strip(), str(c.get("text", ""))
            conf = c.get("confirmed") if c.get("confirmed") in ("pass", "fail") else None
        else:
            label, text = "", str(c)
        label = label or f"contrastive {j + 1}"
        base, n = label, 2
        while label in taken:
            label, n = f"{base} ({n})", n + 1
        taken.add(label)
        out.append({"label": label, "text": text, "confirmed": conf})
    return out


def normalize_confirmed(raw, item):
    """{system: 'pass'|'fail'} of the annotator for the other systems (the reference is always fail)."""
    raw = raw if isinstance(raw, dict) else {}
    return {s: raw[s] for s in item["others"] if raw.get(s) in ("pass", "fail")}


# ---------------------------------------------------------------------------- store

KEEP = object()  # Store.put: keep the saved decision
DECISIONS = ("remove", "undecidable", "finished")


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
        # verdicts of the startup verification for spans that were never saved (only prefilled
        # rules): shown in the page, but not written, so that these spans stay pending
        self.transient = {}

    def view(self, key):
        """The record the page shows: the saved one, else the unsaved startup verification."""
        return self.records.get(key) or self.transient.get(key)

    def write(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for k in self.order + [k for k in self.records if k not in self.order]:
                    if k in self.records:
                        f.write(json.dumps(self.records[k], ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)

    def rules(self, key):
        r = self.records.get(key)
        return r["rules"] if r else self.prefill.get(key, [])

    def put(self, item, rules, contrastive, results=None, model=None, decision=KEEP, confirmed=KEEP):
        """Save rules, contrastive translations, verdicts and the annotator's decision.
        decision: KEEP (the saved one), None, or one of DECISIONS. A 'finished' that the new state
        no longer satisfies (see finish_problems) is dropped back to in progress."""
        old = self.view(item["key"])
        rec = {
            "key": item["key"], "document": item["document"], "audio": item["audio"],
            "seg": item["seg"], "ref_system": item["ref_system"], "span_index": item["span_index"],
            "span": item["span"], "span_start": item["span_start"], "span_end": item["span_end"],
            "source": item["source"], "rules": rules, "contrastive": contrastive,
            "confirmed": (old or {}).get("confirmed", {}) if confirmed is KEEP else confirmed,
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
        self.transient.pop(item["key"], None)
        dec = (old or {}).get("decision") if decision is KEEP else decision
        if dec == "finished" and finish_problems(item, rec):
            dec = None
        rec["decision"] = dec
        if dec:
            rec["decided_at"] = (old or {}).get("decided_at") if dec == (old or {}).get("decision") else rec["saved_at"]
        with self.lock:
            self.records[item["key"]] = rec
        self.write()
        return rec


def startup_verify(items, store, verifier, use_cache=True):
    """Verify the rules of every span (saved, or prefilled from --rules) against all its targets
    before the server starts, in one batch; answers in the cache are not asked again.
    Saved spans get the verdicts written to --output; prefilled-only spans keep them in memory."""
    todo = []
    for it in items:
        rec = store.records.get(it["key"])
        rules = store.rules(it["key"])
        if not rules or (rec or {}).get("decision") == "remove":
            continue
        contrastive = normalize_contrastive((rec or {}).get("contrastive"), it)
        targets = {it["ref_system"]: it["ref_text"], **it["others"],
                   **{c["label"]: c["text"] for c in contrastive if c["text"].strip()}}
        todo.append((it, rec, rules, contrastive, targets))
    if not todo:
        return
    jobs = [(n, s, k) for n, (_, _, rules, _, tg) in enumerate(todo) for s in tg for k in range(len(rules))]
    print(f"startup verification: {len(jobs)} (translation, rule) pairs of {len(todo)} spans",
          file=sys.stderr, flush=True)
    t0 = time.time()

    progress = ProgressLog("startup verification")

    if verifier.dummy:
        answers = ["fail" if s == todo[n][0]["ref_system"] else "pass" for n, s, _ in jobs]
        stats, reasons = {"cached": 0, "asked": len(jobs)}, []
    else:
        chats = [verifier.chat(todo[n][0]["source"], todo[n][4][s], todo[n][2][k]) for n, s, k in jobs]
        try:
            answers, stats, reasons = verifier.answer(chats, use_cache, progress)
        except VerifyError as e:
            print(f"WARNING: startup verification failed, starting without it:\n{e}", file=sys.stderr)
            return
    results = [{s: [None] * len(rules) for s in tg} for _, _, rules, _, tg in todo]
    for (n, s, k), a in zip(jobs, answers):
        v = parse_verdict(a)
        if a is None:
            v["evidence"] = reasons[0] if reasons else "request failed"
        results[n][s][k] = v
    broken = []
    for (it, rec, rules, contrastive, _), res in zip(todo, results):
        summary = summarize(it, rules, res)
        if rec is not None:
            rec.update(verified_with=verifier.label(), results=res, **summary)
            if rec.get("decision") == "finished" and finish_problems(it, rec):
                broken.append(it["key"])
        else:
            store.transient[it["key"]] = {"key": it["key"], "rules": rules, "contrastive": [], "confirmed": {},
                                          "verified_with": verifier.label(),
                                          "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                          "results": res, **summary}
    if store.records:
        store.write()
    errors = sum(v["verdict"] == "error" for res in results for vs in res.values() for v in vs)
    passes = sum(overall(r[it["ref_system"]]) == "pass" for (it, _, rules, _, _), r in zip(todo, results))
    print(f"startup verification done in {time.time() - t0:.0f}s: {stats['cached']} from cache, "
          f"{stats['asked']} asked, {errors} errors; the reference passes all rules in {passes} of "
          f"{len(todo)} spans", file=sys.stderr)
    if reasons:
        print("LLM failures:\n  " + summarize_reasons(reasons).replace("\n", "\n  "), file=sys.stderr)
    if broken:
        print(f"WARNING: {len(broken)} finished span(s) no longer pass the finished check with these "
              f"verdicts (they stay finished; open them to check): {', '.join(broken[:10])}"
              + (" ..." if len(broken) > 10 else ""), file=sys.stderr)


def summarize(item, rules, results):
    harmful = {s: any(v["verdict"] == "fail" for v in vs) for s, vs in results.items()}
    ref = results[item["ref_system"]]
    agree = {s: harmful[s] == bool(h) for s, h in item["same_harm"].items() if s in harmful}
    return {"harmful": harmful,
            "ref_fails_each_rule": [v["verdict"] == "fail" for v in ref],
            "agreement": agree}


def status(rec):
    """pending (never saved), progress (saved or verified), or the annotator's decision."""
    if not rec:
        return "pending"
    return rec.get("decision") or "progress"


def finish_problems(item, rec):
    """Why the span cannot be marked finished yet ([] = it can)."""
    rules = rec.get("rules") or []
    if not rules:
        return ["Write at least one verification rule."]
    res = rec.get("results") or {}
    conf = confirmations(item, rec)
    targets = list(conf)
    out = []
    missing = [t for t in targets if conf[t] is None]
    if missing:
        out.append(f"Choose pass or fail in the human expert column for: {', '.join(missing)}.")
    if not any(conf[t] == "pass" for t in targets):
        out.append("No translation is marked pass in the human expert column. If all systems contain the harm, "
                   "add a contrastive translation without it, mark it pass, and verify.")
    stale = [t for t in targets if len(res.get(t) or []) != len(rules)]
    if stale:
        out.append(f"Verify first: there are no current verdicts for {', '.join(stale)}.")
    done = [t for t in targets if t not in stale]
    unclear = [t for t in done if overall(res[t]) == "error"]
    if unclear:
        out.append(f"Some overall verdicts are ? (LLM error) for {', '.join(unclear)}; verify again.")
    wrong = [f"{t} (LLM judge: {overall(res[t])}, human expert: {conf[t]})" for t in done
             if conf[t] and overall(res[t]) in ("pass", "fail") and overall(res[t]) != conf[t]]
    if wrong:
        out.append("The overall verdict of the LLM judge differs from the human expert for: " + "; ".join(wrong)
                   + ". Change the rules (or your judgement) and verify.")
    return out


def confirmations(item, rec):
    """{translation: 'pass'|'fail'|None}: the reference is fail (the confirmed harm); the other
    systems and the non-empty contrastive translations have the annotator's choice (None = not yet)."""
    sysconf = rec.get("confirmed") or {}
    out = {item["ref_system"]: "fail"}
    for s in item["others"]:
        out[s] = sysconf.get(s)
    for c in rec.get("contrastive") or []:
        if c["text"].strip():
            out[c["label"]] = c.get("confirmed")
    return out


def overall(verdicts):
    """The overall verdict of one translation: fail if it fails a rule, else error if a verdict is
    an error, else pass."""
    vs = [v["verdict"] for v in verdicts]
    return "fail" if "fail" in vs else "error" if "error" in vs else "pass"


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
.it .m{display:inline-block;width:14px;font:700 12.5px ui-monospace,monospace}
.it .br{display:inline-block;width:3.2em;text-align:right;padding-right:.4em;color:var(--mute);font-family:ui-monospace,monospace}
.it .ns{color:var(--fail);font-size:11px;margin-left:4px}
.it.g1{border-top:1px solid var(--line)}.it.gl{border-bottom:1px solid var(--line)}
.multi{color:var(--fail);font-weight:700}
.m.progress{color:var(--err)}.m.finished{color:var(--pass)}.m.remove{color:var(--fail)}.m.undecidable{color:var(--acc)}.m.pending{color:var(--mute)}
.trk{display:flex;gap:4px;margin-right:10px}.trk span{padding:4px 10px;border-radius:14px;font-size:12px;background:var(--hl2);color:var(--mute)}
.trk span.on{background:var(--pass);color:#fff;font-weight:600}
button.dec.on{color:#fff;font-weight:600}button.dec.on[data-d=remove]{background:var(--fail);border-color:var(--fail)}
button.dec.on[data-d=undecidable]{background:var(--acc);border-color:var(--acc)}button.dec.on[data-d=finished]{background:var(--pass);border-color:var(--pass)}
@media (prefers-color-scheme:dark){.trk span.on,button.dec.on{color:#111}}
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
.tt th.ov,.tt td.ov{border-left:2px solid var(--line);width:64px}
.tt td.ov.bad{background:var(--failbg)}
.tt th.ex,.tt td.ex{border-left:2px solid var(--line);border-right:2px solid var(--line);width:84px;text-align:center;white-space:nowrap}
.tt th.ex{font-weight:700;color:var(--ink);font-size:12.5px}.tt th.ex .sub{font-weight:400;color:var(--mute);font-size:11px}
.tt td.ex.fixed{font-weight:700;color:var(--fail);font-size:15px}
.tt th.grp{text-align:center;border-bottom:1px solid var(--line);font-size:11.5px}
button.cf{font-size:13px;font-weight:700;padding:1px 7px;margin:0 1px;color:var(--mute)}
button.cf.pass.on{background:var(--pass);border-color:var(--pass);color:#fff}button.cf.fail.on{background:var(--fail);border-color:var(--fail);color:#fff}
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
#instr{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin-bottom:16px}
#instr summary{cursor:pointer;padding:9px 14px;font-weight:600;user-select:none}
#instr .instr{padding:0 18px 12px;font-size:13.5px;line-height:1.5;max-width:980px}
#instr h3{font-size:14px;margin:12px 0 6px}#instr h4{font-size:13px;margin:12px 0 4px}
#instr li{margin:3px 0}#instr code{font:12.5px ui-monospace,monospace;background:var(--hl2);padding:0 4px;border-radius:3px}
.errbox{margin:10px 0 0;padding:10px 12px;border:1px solid var(--fail);border-radius:8px;background:var(--failbg);color:var(--fail);
 font:12.5px/1.45 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto}
.errbox.note{border-color:var(--err);background:var(--errbg);color:var(--err)}
</style></head><body>
<div id="side"><input id="flt" placeholder="filter: id, span or mark (- ! . x ?)" title="marks: - pending, ! in progress, . finished, x remove, ? undecidable"><div id="list"></div></div>
<div id="main">
<details id="instr" open><summary>Instructions (click to show/hide)</summary><div class="instr">
__INSTRUCTIONS__
</div></details>
 <div class="head"><div><b id="title"></b> <span class="msg" id="pos"></span></div>
  <div class="nav"><div class="trk" id="trk"><span data-s="pending">pending</span><span data-s="progress">in progress</span><span data-s="finished">finished</span></div>
   <button onclick="go(cur-1)">← prev</button><button onclick="go(cur+1)">next →</button></div></div>
 <audio id="aud" controls preload="none"></audio>
 <h2>Source (gold transcript)</h2><div class="box" id="src"></div>
 <h2>Targets and verdicts</h2><div id="tgts"></div>
 <h2>Verification rules <span class="legend">— one rule per line (R1, R2, …); autosaved; Ctrl+Enter = verify</span></h2>
 <textarea id="rules" spellcheck="false"></textarea>
 <div class="bar"><button class="p" id="vbtn" onclick="verify()">Verify</button>
  <span class="msg" id="msg"></span>
  <label class="msg" style="margin-left:auto" id="thinklab"><input type="checkbox" id="think"> reasoning</label>
  <label class="msg" title="ask the LLM again for every pair, and refresh the cache"><input type="checkbox" id="nocache"> ignore cache</label></div>
 <div class="bar"><span class="msg">mark as:</span>
  <button class="dec" data-d="remove" onclick="decide('remove')">remove</button>
  <button class="dec" data-d="undecidable" onclick="decide('undecidable')">undecidable</button>
  <button class="dec" data-d="finished" onclick="decide('finished')">finished</button>
  <span class="msg">(click again to untick)</span></div>
 <pre class="errbox" id="err" hidden></pre>
 <div class="msg" id="vnote"></div>
</div>
<script>
let items=[],cur=0,item=null,dirty=false,contr=[],conf={};  // conf: the annotator's pass/fail per system  // contr: the human contrastive translations being edited
// autosave: every edit calls touch(); 1.5 s after the last edit the rules and contrastive translations are saved.
// edits counts the edits, so that a save that was sent before newer edits does not clear 'dirty';
// seq/applied make sure that an older response never overwrites a newer one.
let edits=0,timer=null,seq=0,applied=0;
const AUTOSAVE_MS=1500;
function touch(){dirty=true;edits++;clearTimeout(timer);timer=setTimeout(autosave,AUTOSAVE_MS)}
function stopAutosave(){clearTimeout(timer);timer=null}
async function autosave(){stopAutosave();if(!dirty||!item)return;const at=edits,s=++seq,i=cur;
 try{const r=await api('/api/save',{i,rules:rulesNow(),contrastive:contr,confirmed:conf});
  if(i!==cur||s<applied)return;applied=s;
  items[cur].status=r.status;item.record=r.record;renderList();renderStatus();   // no re-render: keep the cursor where it is
  if(edits===at)dirty=false   // silent: only a failure is shown
  if(r.notice)showErr(r.notice)}
 catch(e){$('msg').textContent='autosave failed';showErr('Autosave failed: '+e.message);timer=setTimeout(autosave,10000)}}
const $=id=>document.getElementById(id), esc=s=>(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const MARK={pending:'-',progress:'!',finished:'.',remove:'x',undecidable:'?'};
function renderStatus(){const st=items[cur].status,ph=st=='pending'?'pending':st=='progress'?'progress':'finished';
 document.querySelectorAll('#trk span').forEach(e=>{e.classList.toggle('on',e.dataset.s==ph);
  if(e.dataset.s=='finished')e.textContent=ph=='finished'&&st!='finished'?'finished: '+st:'finished'});
 document.querySelectorAll('button.dec').forEach(b=>b.classList.toggle('on',b.dataset.d==st))}
function applyResp(r,s){if(s!=null){if(s<applied)return;applied=s}contr=r.record.contrastive.map(c=>({...c}));conf={...(r.record.confirmed||{})};items[cur].status=r.status;item.record=r.record;
 renderRes(r.record);renderList();renderStatus()}
async function decide(d){const st=items[cur].status,nd=st==d?null:d;showErr('');stopAutosave();const s=++seq;
 try{const r=await api('/api/decide',{i:cur,rules:rulesNow(),contrastive:contr,decision:nd,confirmed:conf});dirty=false;applyResp(r,s);
  $('msg').textContent=nd?'marked as '+nd:'unticked: in progress';
  if(nd=='finished'&&GROUP[cur].length>1&&!r.notice)showErr("There are multiple spans in this segment. Check that the finished rules don't check the same span. If they do, remove one of them.",true);
  else showErr(r.notice,nd==null)}
 catch(e){$('msg').textContent='not marked';showErr(e.message);if(e.data&&e.data.record){dirty=false;applyResp(e.data)}}}
async function api(p,body){let r;
 try{r=await fetch(p,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})}
 catch(e){throw new Error('The editor server is not reachable ('+e.message+'). Is rule_editor.py still running?')}
 const t=await r.text();let j=null;try{j=JSON.parse(t)}catch(e){}
 if(!r.ok){const e=new Error(j&&j.error?j.error:`HTTP ${r.status}: ${t}`);e.data=j;throw e}return j}
// "reasoning" tickbox: shows the --enable-thinking default until clicked; then ticked = low, unticked = no
let thinkTouched=false,thinkDefault=null;
function initThinking(d){if(thinkTouched)return;thinkDefault=d;$('think').checked=d!=='no';
 $('thinklab').title='ticked: reasoning low; unticked: no reasoning. Until you click it, the command-line setting is used: '
  +(d?'--enable-thinking '+d:'the model\'s default (no --enable-thinking)')}
function thinkingNow(){return thinkTouched?($('think').checked?'low':'no'):null}
function showErr(m,note){const b=$('err');b.textContent=m||'';b.hidden=!m;b.classList.toggle('note',!!note)}
// spans of the same segment: GROUP[i] = [indices of all items of that segment], in order
let GROUP=[];
function buildGroups(){const by={};items.forEach((it,i)=>(by[it.seg]=by[it.seg]||[]).push(i));GROUP=items.map(it=>by[it.seg])}
function renderList(){const f=$('flt').value.toLowerCase();
 $('list').innerHTML=items.map((it,i)=>{const t=`${MARK[it.status]} ${it.seg} ${it.span}`.toLowerCase();
  if(f&&!t.includes(f))return'';
  const g=GROUP[i],k=g.indexOf(i),n=g.length,m=`<span class="m ${it.status}">${MARK[it.status]}</span>`;
  const cls='it'+(i==cur?' cur':'')+(n>1&&k==0?' g1':'')+(n>1&&k==n-1?' gl':'');
  // first span of a segment: id + span (and how many spans); the others branch off below it
  const body=k==0?`${it.seg} · ${esc(it.span)}${n>1?`<span class="ns">${n} spans</span>`:''}`
   :`<span class="br">${k==n-1?'└─':'├─'}</span>${esc(it.span)}`;
  return `<div class="${cls}" onclick="go(${i})" title="${esc(it.seg+' · '+it.span)}">${m}${body}</div>`}).join('')}
function hl(text,s,e,others){ // highlight current span [s,e) and other confirmed spans
 const marks=[];if(s!=null)marks.push([s,e,'']);(others||[]).forEach(o=>marks.push([o[0],o[1],'o']));
 marks.sort((a,b)=>a[0]-b[0]);let out='',p=0;
 for(const [a,b,c] of marks){if(a<p)continue;out+=esc(text.slice(p,a))+`<mark class="${c}">${esc(text.slice(a,b))||'∅'}</mark>`;p=b}
 return out+esc(text.slice(p))}
async function go(i){if(i<0||i>=items.length)return;
 if(dirty){await autosave();if(dirty&&!confirm('Your changes could not be saved (see the error). Leave anyway?'))return}
 cur=i;item=await api('/api/item?i='+i);dirty=false;location.hash=i;
 {const g=GROUP[i],n=g.length,k=g.indexOf(i)+1;
  $('title').innerHTML=esc(item.seg+' · '+item.ref_system+' · ')+(n>1?`<span class="multi">span ${k} / ${n}</span>`:'span 1 / 1')}$('pos').textContent=`${i+1} / ${items.length}`;
 $('src').textContent=item.source;
 const a=$('aud');a.pause();if(item.has_audio){a.src='/api/audio?i='+i;a.style.display=''}else{a.removeAttribute('src');a.style.display='none'}
 contr=((item.record&&item.record.contrastive)||[]).map(c=>({...c}));conf={...((item.record&&item.record.confirmed)||{})};
 $('rules').value=(item.rules||[]).join('\n');$('msg').textContent='';showErr(item.backend_error);initThinking(item.default_thinking);renderStatus();renderRes(item.record);renderList();
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
  h+=`<td class="r ${c}" title="${esc('R'+(k+1)+': '+rec.rules[k]+'\n\n'+v.verdict.toUpperCase()+(v.evidence?': '+v.evidence:'')+(v.raw?'\n\nraw: '+v.raw:''))}">${g}</td>`}
 return h}
function overallOf(vs){if(!vs)return null;const v=vs.map(x=>x.verdict);
 return v.includes('fail')?'fail':v.includes('error')?'error':'pass'}
const GLYPH={pass:['ok','✓'],fail:['no','✗'],error:['er','?']};
function confCell(s,j){ // the "confirmed" column: fixed fail for the reference, pass/fail buttons for the others
 if(s===item.ref_system)return `<td class="ex fixed" title="the confirmed harmful error: always fail">✗</td>`;
 const v=j==null?conf[s]:contr[j].confirmed,a=j==null?`data-s="${esc(s)}"`:`data-j="${j}"`;
 return `<td class="ex">`+['pass','fail'].map(x=>`<button class="cf ${x}${v===x?' on':''}" ${a} data-v="${x}" `
  +`title="human expert: ${x}${v===x?' (click again to clear)':''}">${GLYPH[x][1]}</button>`).join('')+`</td>`}
function overallCell(rec,s,c,expect){ // overall verdict of the LLM judge (red if it differs from the human expert)
 const o=overallOf(savedFor(rec,s,c)),ex='';
 if(!o)return `<td class="r ov msg">·</td>`;
 const [k,g]=GLYPH[o],bad=expect&&o!==expect;
 return `<td class="r ov ${k}${bad?' bad':''}" title="${o=='fail'?'fails at least one rule':o=='pass'?'passes all rules':'a verdict is an LLM error'}`
  +`${expect?'; human expert: '+expect:''}">${g}</td>`+ex}
document.addEventListener('click',e=>{const b=e.target.closest('button.cf');if(!b)return;
 const v=b.dataset.v;
 if(b.dataset.j!=null){const c=contr[+b.dataset.j];c.confirmed=c.confirmed===v?null:v}
 else{const s=b.dataset.s;if(conf[s]===v)delete conf[s];else conf[s]=v}
 touch();renderRes(item.record)});
function nRules(){return rulesNow().length||(item.record&&item.record.rules?item.record.rules.length:0)}
function renderRes(rec){ // table: system | target | one tick per rule; human contrastive rows at the bottom
 const sys=[item.ref_system,...Object.keys(item.others)],n=nRules();
 let t='<table class="tt"><tr><th rowspan="2">system</th><th rowspan="2">target</th>'
  +'<th rowspan="2" class="ex">human expert<br><span class="sub">(your judgement)</span></th>'
  +`<th colspan="${n+1}" class="grp">LLM judge with your rules</th></tr><tr>`
  +Array.from({length:n},(_,k)=>`<th class="r">R${k+1}</th>`).join('')+'<th class="r ov">overall</th></tr>';
 for(const s of sys)t+=`<tr><td class="sys">${esc(s)}</td><td>${s==item.ref_system?hl(item.ref_text,item.span_start,item.span_end,item.other_spans):esc(item.others[s])}</td>${confCell(s,null)}${ticks(rec,s,null,n)}${overallCell(rec,s,null,s==item.ref_system?'fail':conf[s])}</tr>`;
 contr.forEach((c,j)=>{t+=`<tr><td class="sys"><input class="cl" data-j="${j}" value="${esc(c.label)}" title="label (editable)"><br><button class="x" onclick="delContr(${j})" title="remove this contrastive translation">remove</button></td>`
   +`<td><textarea class="ct" data-j="${j}" rows="2" placeholder="human contrastive translation">${esc(c.text)}</textarea></td>${confCell(null,j)}${ticks(rec,null,c,n)}${overallCell(rec,null,c,c.confirmed)}</tr>`});
 t+=`<tr><td colspan="${4+n}"><button onclick="addContr()">+ add contrastive translation</button></td></tr>`;
 $('tgts').innerHTML=t+'</table>';
 document.querySelectorAll('.ct').forEach(grow);
 let note='';
 if(rec&&rec.results){const ro=overallOf(savedFor(rec,item.ref_system,null));
  note=(ro=='pass'?`<span class="warn">${esc(item.ref_system)} passes all rules: they do not catch the confirmed error.</span> `:'')
   +`Verified with ${esc(rec.verified_with)} at ${esc(rec.saved_at)}.`+(freshFor(rec,item.ref_system,null)?'':' <span class="warn">Rules edited since; verify again.</span>')}
 $('vnote').innerHTML=note}
function grow(el){el.style.height='auto';el.style.height=(el.scrollHeight+2)+'px'}
function freeLabel(){const used=new Set([item.ref_system,...Object.keys(item.others),...contr.map(c=>c.label)]);
 let k=1;while(used.has('contrastive '+k))k++;return 'contrastive '+k}
function addContr(){contr.push({label:freeLabel(),text:''});touch();renderRes(item.record);const a=document.querySelectorAll('.ct');a[a.length-1].focus()}
function delContr(j){if(contr[j].text.trim()&&!confirm('Remove '+contr[j].label+'?'))return;contr.splice(j,1);touch();renderRes(item.record)}
document.addEventListener('input',e=>{const el=e.target,j=+el.dataset.j;
 if(el.classList.contains('cl')){contr[j].label=el.value;touch();return}   // renaming keeps the verdicts
 if(!el.classList.contains('ct'))return;
 contr[j].text=el.value;touch();grow(el);
 el.closest('tr').querySelectorAll('td.r').forEach(td=>{td.className='r msg';td.textContent='·';td.removeAttribute('title')})});
const rulesNow=()=>$('rules').value.split('\n').map(s=>s.trim()).filter(Boolean);
async function save(){showErr('');stopAutosave();const s=++seq;try{const r=await api('/api/save',{i:cur,rules:rulesNow(),contrastive:contr,confirmed:conf});dirty=false;applyResp(r,s);$('msg').textContent='saved';showErr(r.notice)}catch(e){$('msg').textContent='not saved';showErr(e.message)}}
async function verify(){const rules=rulesNow();if(!rules.length){$('msg').textContent='write at least one rule';return}
 $('vbtn').disabled=true;showErr('');const t0=Date.now();$('msg').textContent='verifying '+rules.length+' rule(s) × '+(1+Object.keys(item.others).length+contr.filter(c=>c.text.trim()).length)+' targets…';
 stopAutosave();const s=++seq;
 try{const r=await api('/api/verify',{i:cur,rules,contrastive:contr,no_cache:$('nocache').checked,thinking:thinkingNow(),confirmed:conf});dirty=false;applyResp(r,s);
  const st=r.stats||{};$('msg').textContent=`verified and saved (${((Date.now()-t0)/1000).toFixed(1)} s; ${st.cached||0} from cache, ${st.asked||0} asked)`+(r.errors&&r.errors.length?', with errors':'');
  showErr([...(r.errors||[]),r.notice].filter(Boolean).join('\n\n'))}catch(e){$('msg').textContent='verification failed';showErr(e.message)}
 $('vbtn').disabled=false}
$('rules').addEventListener('input',()=>{touch();if(item)renderRes(item.record)});$('flt').addEventListener('input',renderList);
$('think').addEventListener('change',()=>{thinkTouched=true});
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key=='Enter'){e.preventDefault();verify()}
 if((e.ctrlKey||e.metaKey)&&e.key=='s'){e.preventDefault();save()}  // save at once (and no browser "save page" dialog)
 if(e.altKey&&e.key=='ArrowDown'){e.preventDefault();go(cur+1)}if(e.altKey&&e.key=='ArrowUp'){e.preventDefault();go(cur-1)}});
// closing or reloading the tab: send the last edits without waiting for an answer
window.addEventListener('pagehide',()=>{if(dirty&&item)navigator.sendBeacon('/api/save',
 new Blob([JSON.stringify({i:cur,rules:rulesNow(),contrastive:contr,confirmed:conf})],{type:'application/json'}))});
window.addEventListener('hashchange',()=>{const i=parseInt(location.hash.slice(1));if(!isNaN(i)&&i!==cur)go(i)});
try{if(localStorage.getItem('instrOpen')==='0')$('instr').open=false}catch(e){}
$('instr').addEventListener('toggle',()=>{try{localStorage.setItem('instrOpen',$('instr').open?'1':'0')}catch(e){}});
(async()=>{items=await api('/api/items');buildGroups();go(Math.min(parseInt(location.hash.slice(1))||0,items.length-1))})();
</script></body></html>"""


def dropped(old, rec, item, chosen=KEEP):
    """A message if the span was finished and is not any more. chosen: the decision the annotator
    clicked (KEEP if the change came from an edit, i.e. 'finished' was dropped automatically)."""
    if old != "finished" or rec.get("decision") == "finished":
        return None
    if chosen is None:
        return 'You unticked "finished"; you may have something to finish.'
    if chosen is not KEEP:  # switched to remove / undecidable: nothing to say
        return None
    probs = finish_problems(item, rec)
    return "No longer finished (back to in progress)" + (":\n- " + "\n- ".join(probs) if probs else ".")


def read_instructions(path):
    """The instructions fragment, read on every page load, so edits show after a reload."""
    if not path:
        return "<p class=\"msg\">No instructions (started with --instructions '').</p>"
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"<p class=\"warn\">Cannot read the instructions file {html_escape(path)}: {html_escape(str(e))}</p>"


def html_escape(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_handler(items, store, verifier, audio_roots, instructions=None):
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
            self.send_header("Cache-Control", "no-store")  # never run an old copy of the page
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
                    self._send(200, PAGE.replace("__INSTRUCTIONS__", read_instructions(instructions)), "text/html")
                elif u.path == "/api/items":
                    self._json([{"seg": it["seg"], "span": it["span"],
                                 "status": status(store.records.get(it["key"]))} for it in items])
                elif u.path == "/api/audio":
                    self._audio(audio_path(self._item(int(parse_qs(u.query)["i"][0]))))
                elif u.path == "/api/item":
                    it = self._item(int(parse_qs(u.query)["i"][0]))
                    rec = store.view(it["key"])
                    if rec:  # older files: contrastive as plain strings
                        rec["contrastive"] = normalize_contrastive(rec.get("contrastive"), it)
                    self._json({**it, "rules": store.rules(it["key"]), "backend_error": verifier.init_error, "default_thinking": verifier.thinking, "has_audio": bool(audio_path(it)),
                                "record": store.view(it["key"])})
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
                confirmed = normalize_confirmed(body["confirmed"], it) if "confirmed" in body else KEEP
                old = (store.records.get(it["key"]) or {}).get("decision")
                if self.path == "/api/save":
                    rec = store.put(it, rules, contrastive, confirmed=confirmed)
                elif self.path == "/api/decide":
                    d = body.get("decision") or None
                    if d not in (None, *DECISIONS):
                        raise ValueError(f"unknown decision {d!r}")
                    rec = store.put(it, rules, contrastive, decision=d, confirmed=confirmed)
                    if d == "finished" and rec["decision"] != "finished":
                        probs = finish_problems(it, rec)
                        return self._json({"error": "Cannot mark as finished:\n- " + "\n- ".join(probs),
                                           "record": rec, "status": status(rec)}, 409)
                elif self.path == "/api/verify":
                    t0 = time.time()
                    # the "reasoning" tickbox: low or no; absent/null = the --enable-thinking default
                    thinking = body.get("thinking") or DEFAULT
                    if thinking is not DEFAULT and thinking not in THINKING_LEVELS:
                        raise ValueError(f"unknown thinking level {thinking!r}")
                    try:
                        results, errors, stats = verify_item(verifier, it, rules, contrastive,
                                                             use_cache=not body.get("no_cache"),
                                                             thinking=thinking)
                    except VerifyError as e:
                        return self._json({"error": str(e)}, 502)
                    rec = store.put(it, rules, contrastive, results, verifier.label(thinking), confirmed=confirmed)
                    print(f"{it['key']}: {len(rules)} rule(s) verified in {time.time() - t0:.1f}s "
                          f"({stats['cached']} cached, {stats['asked']} asked)", file=sys.stderr)
                    return self._json({"record": rec, "status": status(rec), "errors": errors, "stats": stats,
                                       "notice": dropped(old, rec, it)})
                else:
                    return self._send(404, "not found", "text/plain")
                chosen = (body.get("decision") or None) if self.path == "/api/decide" else KEEP
                self._json({"record": rec, "status": status(rec), "notice": dropped(old, rec, it, chosen)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send(500, f"{type(e).__name__}: {e}", "text/plain")

    return H


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="confirmed harmful spans, JSONL (targets with harm_spans)")
    p.add_argument("--systems", help="JSONL with the translations of all systems per segment "
                                     "('text', 'systems_info'); matched to the data by document and audio. "
                                     "Without it, the other systems come from 'other_outputs' in the data")
    p.add_argument("--output", "-o", required=True, help="JSONL with rules and verdicts (loaded if it exists)")
    p.add_argument("--rules", help="prefill rules from a verification_rules.jsonl")
    p.add_argument("--prompt", help="verifier prompt file with {rule}, {source_text}, {translation} "
                                    "(default: the built-in prompt of llm_verifier.py)")
    p.add_argument("--audio-root", action="append", default=[],
                   help="directory the 'audio' paths are relative to (repeatable; tried after the "
                        "current directory and the directory of the data file)")
    p.add_argument("--cache", help="disk cache of LLM answers (default: <output>.llm-cache.jsonl)")
    p.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    p.add_argument("--no-startup-verify", action="store_true",
                   help="do not verify all rules (saved and --rules) with the LLM before the server starts")
    p.add_argument("--instructions", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "instructions.html"),
                   help="HTML fragment shown at the top of the page, read on every page load "
                        "(default: instructions.html next to rule_editor.py; '' = none)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    add_thinking_argument(p)
    llm_backend.add_arguments(p)
    args = p.parse_args()

    items = load_items(args.data, load_systems(args.systems) if args.systems else None)
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
    verifier = Verifier(llm_backend.BackendConfig.from_args(args), read_prompt(args.prompt),
                        None if args.no_cache else (args.cache or os.path.splitext(args.output)[0] + ".llm-cache.jsonl"),
                        thinking=args.enable_thinking)
    if not args.no_startup_verify:
        startup_verify(items, store, verifier, use_cache=not args.no_cache)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(items, store, verifier,
                                                         args.audio_root + [os.path.dirname(os.path.abspath(args.data))],
                                                         args.instructions))
    print(f"open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}",
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()