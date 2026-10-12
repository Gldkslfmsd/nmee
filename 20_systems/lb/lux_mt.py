#!/usr/bin/env python3
"""
Translate the "segment" field of JSONL documents from Luxembourgish (lb)
using the LuxMT API at https://luxasr.uni.lu/luxmt/translate.

Usage:
    python luxmt_translate.py doc1.lb.jsonl doc2.lb.jsonl -o outdir [-t en]

For each input file X.lb.jsonl, writes outdir/X.<tgt>.jsonl with one
{"segment": <translation>, "start": ..., "end": ...} records, in order.
Word-level fields are dropped. Normally there is one record per input
segment. If a --chunk translation comes back with a different number of
lines than it was sent, the whole chunk becomes one record, with the start
of its first segment and the end of its last. Empty translations are skipped.

--continue resumes interrupted runs: it finds the last written record (by its
"end" timestamp) and translates only the input segments after it.
"""

import argparse
import json
import logging
import os
import sys
import time

import requests

API_URL = "https://luxasr.uni.lu/luxmt/translate"
SOURCE_LANG = "lb"
FIELD = "segment"

log = logging.getLogger("luxmt")


def extract_translation(data):
    """Pull the translated text out of the API response, whatever its exact shape."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("translation", "translated_text", "translatedText",
                    "output", "result", "text"):
            if key in data:
                return extract_translation(data[key])
    if isinstance(data, list) and data:
        return extract_translation(data[0])
    raise ValueError(f"Unrecognised API response: {data!r}")


def translate(session, text, target_lang, plain_language=False,
              retries=3, timeout=120):
    if not text or not text.strip():
        return text
    payload = {
        "text": text,
        "source_lang": SOURCE_LANG,
        "target_lang": target_lang,
        "summary": False,
        "plain_language": plain_language,
    }
    for attempt in range(1, retries + 1):
        try:
            r = session.post(API_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                return extract_translation(r.json())
            except json.JSONDecodeError:
                return r.text.strip()
        except (requests.RequestException, ValueError) as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            log.warning("request failed (%s); retrying in %ds", e, wait)
            time.sleep(wait)


def output_path(inp, target_lang, outdir):
    base = os.path.basename(inp)
    for suf in (f".{SOURCE_LANG}.jsonl", ".jsonl"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return os.path.join(outdir, f"{base}.{target_lang}.jsonl")


def make_chunks(texts, chunk_size):
    """Group consecutive segment indices so that the newline-joined text fits
    into chunk_size characters. Segments that contain a newline themselves, or
    are longer than chunk_size, go into a chunk of their own."""
    chunks, cur, cur_len = [], [], 0
    for i, t in enumerate(texts):
        alone = chunk_size <= 0 or "\n" in t or len(t) > chunk_size
        add_len = len(t) + (1 if cur else 0)
        if cur and (alone or cur_len + add_len > chunk_size):
            chunks.append(cur)
            cur, cur_len = [], 0
        if alone:
            chunks.append([i])
            continue
        cur.append(i)
        cur_len += len(t) + (1 if len(cur) > 1 else 0)
    if cur:
        chunks.append(cur)
    return chunks


def translate_chunk(session, texts, args):
    """Translate a list of segments in one request and split the output back
    by lines. Returns (translations, aligned). On a line-count mismatch, warn
    and return the whole chunk translation as one merged string."""
    kw = dict(plain_language=args.plain_language,
              retries=args.retries, timeout=args.timeout)
    if len(texts) == 1:
        return [translate(session, texts[0], args.target_lang, **kw)], True
    out = translate(session, "\n".join(texts), args.target_lang, **kw)
    lines = out.strip("\n").split("\n")
    n = len(texts)
    if len(lines) == n:
        return lines, True
    log.warning("chunk of %d segments came back as %d lines; "
                "merging them into one output segment", n, len(lines))
    return [" ".join(l.strip() for l in lines if l.strip())], False


def load_docs(inp):
    docs = []
    with open(inp, encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if FIELD not in doc:
                log.warning("%s:%d: no '%s' field, skipping", inp, lineno, FIELD)
                continue
            docs.append(doc)
    return docs


def resume_point(out, docs):
    """Return the index of the first input segment not yet in the output file.
    Drops a trailing partially written line, if any."""
    if not os.path.exists(out):
        return 0
    recs, good_bytes = [], 0
    with open(out, "rb") as f:
        for raw in f:
            try:
                recs.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not raw.endswith(b"\n"):
                recs.pop()
                break
            good_bytes += len(raw)
    if good_bytes < os.path.getsize(out):
        log.warning("%s: dropping a partial last line", out)
        with open(out, "r+b") as f:
            f.truncate(good_bytes)
    if not recs:
        return 0
    last_end = recs[-1].get("end")
    if last_end is not None:
        for i in range(len(docs) - 1, -1, -1):
            if docs[i].get("end") == last_end:
                return i + 1
        log.warning("%s: last end=%s not found in input; resuming by record count",
                    out, last_end)
    return min(len(recs), len(docs))


def fmt_dur(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Progress:
    """Tracks request times and estimates the remaining time from characters."""

    def __init__(self, total_chars):
        self.total = total_chars
        self.done = 0
        self.req_time = 0.0
        self.n_req = 0

    def update(self, chars, seconds):
        self.done += chars
        self.req_time += seconds
        self.n_req += 1

    def eta(self):
        if not self.done:
            return "?"
        return fmt_dur((self.total - self.done) * self.req_time / self.done)

    def pct(self):
        return 100.0 * self.done / self.total if self.total else 100.0


def process_file(session, job, args, progress):
    inp, out, docs, first = job["inp"], job["out"], job["docs"], job["first"]
    texts = [d[FIELD] or "" for d in docs]
    todo = list(range(first, len(docs)))
    chunks = [[todo[j] for j in c] for c in
              make_chunks([texts[i] for i in todo],
                          args.max_chunk_size if args.chunk else 0)]
    if not chunks:
        log.info("%s: already complete (%d segments), skipping", out, len(docs))
        return

    log.info("%s -> %s: %d segments, %d to do in %d requests%s",
             inp, out, len(docs), len(todo), len(chunks),
             f" (resuming at segment {first + 1})" if first else "")
    t_file = time.time()
    with open(out, "a" if first else "w", encoding="utf-8") as fout:
        for ci, idxs in enumerate(chunks, 1):
            chunk_texts = [texts[i] for i in idxs]
            n_chars = sum(len(t) for t in chunk_texts)
            t0 = time.time()
            translations, aligned = translate_chunk(session, chunk_texts, args)
            dt = time.time() - t0
            progress.update(n_chars, dt)
            log.info("%s: request %d/%d, segments %d-%d/%d, %d chars, %.2fs | "
                     "total %.1f%%, ETA %s",
                     os.path.basename(inp), ci, len(chunks), idxs[0] + 1,
                     idxs[-1] + 1, len(docs), n_chars, dt, progress.pct(),
                     progress.eta())
            if aligned:
                groups = [([i], tr) for i, tr in zip(idxs, translations)]
            else:
                groups = [(idxs, translations[0])]
            for g, tr in groups:
                if not tr.strip():
                    continue
                rec = {FIELD: tr}
                if "start" in docs[g[0]]:
                    rec["start"] = docs[g[0]]["start"]
                if "end" in docs[g[-1]]:
                    rec["end"] = docs[g[-1]]["end"]
                print(json.dumps(rec, ensure_ascii=False), file=fout, flush=True)
    log.info("%s: done in %s", out, fmt_dur(time.time() - t_file))


def main():
    p = argparse.ArgumentParser(description="Translate JSONL segments from lb with LuxMT.")
    p.add_argument("files", nargs="+", help="input .lb.jsonl files")
    p.add_argument("-t", "--target-lang", default="en", help="target language (default: en)")
    p.add_argument("-o", "--outdir", required=True, help="output directory")
    p.add_argument("--chunk", action="store_true",
                   help="join as many consecutive segments as fit into "
                        "--max-chunk-size characters with newlines and translate them "
                        "in one request (default: one segment per request)")
    p.add_argument("--max-chunk-size", type=int, default=5000,
                   help="max characters per chunk with --chunk (default: 5000)")
    p.add_argument("--continue", dest="cont", action="store_true",
                   help="resume: keep existing output files and translate only the "
                        "segments after the last one written (skip complete files)")
    p.add_argument("--plain-language", action="store_true",
                   help="set plain_language=true in the request")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120, help="per-request timeout in seconds")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(args.outdir, exist_ok=True)

    jobs = []
    for inp in args.files:
        docs = load_docs(inp)
        out = output_path(inp, args.target_lang, args.outdir)
        first = resume_point(out, docs) if args.cont else 0
        jobs.append(dict(inp=inp, out=out, docs=docs, first=first))
    total_chars = sum(len(j["docs"][i][FIELD] or "")
                      for j in jobs for i in range(j["first"], len(j["docs"])))
    log.info("%d files, %d characters to translate", len(jobs), total_chars)

    progress = Progress(total_chars)
    t_all = time.time()
    with requests.Session() as session:
        for job in jobs:
            process_file(session, job, args, progress)
    log.info("all done: %d requests in %s (%.2fs/request on average)",
             progress.n_req, fmt_dur(time.time() - t_all),
             progress.req_time / progress.n_req if progress.n_req else 0)


if __name__ == "__main__":
    main()