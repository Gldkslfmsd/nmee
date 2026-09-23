#!/usr/bin/env python3
"""Aligning a system's text to existing sentences, for add_and_align_sentences.py.

align(method, words, segments, recs, ref_system, opts) returns (sent_of, labels):
  sent_of[i] = index of the sentence that word i belongs to (monotonic)
  labels[k]  = how sentence k was aligned, e.g. "mwer", "time", "vecalign 2:1 split"

Methods:
  mwer      Same language as ref_system. The added words are re-segmented to the sentences of
            ref_system with mweralign (https://github.com/mjpost/mweralign; Post & Hoang 2025,
            AS-WER of Matusov et al. 2005). No timestamps needed.
  vecalign  Different language. The added text is split into sentences (Moses), aligned to the
            sentences of ref_system with the original Vecalign (Thompson & Koehn 2019;
            https://github.com/thompsonb/vecalign) over multilingual sentence embeddings (LaBSE), and
            an n:m unit is then divided among its n sentences (existing sentences are never merged).
  time      Word timestamps only: every word goes to the sentence it overlaps most.

Vecalign and mweralign are used from the git submodules under third_party/ (see setup_aligners.sh) or
from installed packages of the same name. Models are loaded at most once per process.
"""
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path

# third_party/ next to this file or in the working directory; an installed package is the fallback
THIRD_PARTY = [Path(__file__).resolve().parent / "third_party", Path("third_party")]

_SUBMODULES = {}
_ENCODERS = {}
_SEGMENTERS = {}


@dataclass
class Options:
    """Everything the aligners can be tuned with; Options.from_args() fills it from argparse."""
    lan: str = None
    # mwer
    mwer_tokenizer: str = "spm32k"
    mweralign_dir: str = None
    # vecalign
    tgt_split: str = "moses"
    embed_model: str = "sentence-transformers/LaBSE"
    embed_device: str = None
    embed_fp16: bool = False
    embed_batch_size: int = 128
    max_size: int = 4
    search_buffer_size: int = 5
    max_size_full_dp: int = 300
    del_percentile: float = 0.2
    vecalign_dir: str = None
    # splitting an n:m unit among its source sentences
    split_method: str = "embed"
    split_window: int = 20
    split_length_weight: float = 0.1
    split_punct_bonus: float = 0.03

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)})


def has_times(words):
    return bool(words) and all("start" in w and "end" in w for w in words)


def align(method, words, segments, recs, ref_system, opts):
    """Dispatch to one of the three methods. `segments` are the added file's own segments (used by
    --tgt-split segments), `recs` the existing sentences, `ref_system` the system to align to."""
    if method == "time":
        if not has_times(words):
            raise ValueError("--align time needs word timestamps; use --align mwer or vecalign")
        return align_by_time(words, recs)
    if method == "mwer":
        return align_by_mwer(words, recs, ref_system, opts)
    if method == "vecalign":
        return align_by_vecalign(words, segments, recs, ref_system, opts)
    raise ValueError(f"unknown alignment method {method!r}")


def use_submodule(name, extra_dir=None):
    """Put the submodule's copy of `name` first on sys.path, if it is present and built (cached)."""
    key = (name, extra_dir)
    if key in _SUBMODULES:
        return _SUBMODULES[key]
    roots = ([Path(extra_dir)] if extra_dir else []) + [d / name for d in THIRD_PARTY]
    found = None
    for root in roots:
        for path in (root, root / "python"):  # vecalign: repo root; mweralign: repo/python
            if (path / name / "__init__.py").exists():
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
                found = str(path)
                break
        if found:
            break
    _SUBMODULES[key] = found
    return found


# ---------------------------------------------------------------- time

def align_by_time(words, recs):
    """Sentence index per word: max time overlap, else nearest; monotonic."""
    begs = [r["beg"] for r in recs]
    ends = [r["end"] for r in recs]
    out, j, n_gap = [], 0, 0
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        best, best_ov = None, 0.0
        lo = max(0, j - 2)
        k = lo
        while k < len(recs) and begs[k] <= we + 1e-9:
            ov = min(we, ends[k]) - max(ws, begs[k])
            if ov > best_ov:
                best, best_ov = k, ov
            k += 1
        if best is None:
            mid = (ws + we) / 2
            best = min(range(lo, len(recs)),
                       key=lambda q: 0 if begs[q] <= mid <= ends[q]
                       else min(abs(mid - begs[q]), abs(mid - ends[q])))
            n_gap += 1
        best = max(best, j)
        j = best
        out.append(best)
    if n_gap:
        print(f"time: {n_gap} word(s) outside any sentence, assigned to the nearest", file=sys.stderr)
    return out, ["time"] * len(recs)


# ---------------------------------------------------------------- mwer (mweralign)

def _mwer_segmenter(opts):
    """mweralign tokenizer, loaded at most once per process."""
    if opts.mwer_tokenizer in _SEGMENTERS:
        return _SEGMENTERS[opts.mwer_tokenizer]
    seg = None
    if opts.mwer_tokenizer == "cj":
        from mweralign.segmenter import CJSegmenter
        seg = CJSegmenter()
    elif opts.mwer_tokenizer not in ("none", "whitespace"):
        from mweralign import models
        from mweralign.segmenter import SPSegmenter
        seg = SPSegmenter(models.resolve(opts.mwer_tokenizer))
    _SEGMENTERS[opts.mwer_tokenizer] = seg
    return seg


def align_by_mwer(words, recs, ref_system, opts):
    """Re-segment the added words to the sentences of ref_system with mweralign, then map the output
    lines back to the added words (by characters, so word timestamps survive)."""
    src = use_submodule("mweralign", opts.mweralign_dir)
    try:
        from mweralign import align_texts
        from mweralign.segmenter import SPSegmenter
    except ImportError:
        sys.exit("mweralign not found: run ./setup_aligners.sh or pip install mweralign")
    if src and not getattr(align_by_mwer, "_said", False):
        print(f"mwer: using mweralign from {src}", file=sys.stderr)
        align_by_mwer._said = True

    segmenter = _mwer_segmenter(opts)
    is_tokenized = isinstance(segmenter, SPSegmenter) and opts.lan not in ("zh", "ja")

    def tok(text):
        text = " ".join(text.split())
        return " ".join(segmenter.encode(text)) if segmenter is not None else text

    # an empty reference line would change the number of segments, so it gets a placeholder
    refs = [tok(r["text"].get(ref_system, "").strip()) or "_" for r in recs]
    hyp = tok(" ".join(str(w["word"]).strip() for w in words))
    result = align_texts("\n".join(refs), hyp, is_tokenized=is_tokenized,
                         forbid_midword_boundary=is_tokenized)
    lines = result.split("\n")
    if len(lines) > len(recs) and not "".join(lines[len(recs):]).strip():
        lines = lines[:len(recs)]
    if len(lines) != len(recs):
        raise ValueError(f"mweralign returned {len(lines)} lines for {len(recs)} sentences")
    if segmenter is not None:
        lines = [segmenter.decode(l) for l in lines]

    # each word goes to the line of its first character (whitespace ignored)
    line_of_char = [k for k, l in enumerate(lines) for c in l if not c.isspace()]
    n_chars = sum(len([c for c in str(w["word"]) if not c.isspace()]) for w in words)
    if len(line_of_char) != n_chars:
        print("WARNING: mweralign output differs from the input characters; the word mapping may be off "
              "(try --mwer-tokenizer none)", file=sys.stderr)
    sent_of, pos, cur = [], 0, 0
    for w in words:
        k = line_of_char[pos] if pos < len(line_of_char) else len(recs) - 1
        cur = max(k, cur)  # keep monotonic
        sent_of.append(cur)
        pos += len([c for c in str(w["word"]) if not c.isspace()])
    print(f"mwer: mweralign ({opts.mwer_tokenizer}) re-segmented {len(words)} words into "
          f"{len(recs)} sentences", file=sys.stderr)
    return sent_of, ["mwer"] * len(recs)


# ---------------------------------------------------------------- vecalign

def _split_target_sentences(words, segments, opts):
    """Target sentences as lists of word indices."""
    if opts.tgt_split == "segments":
        return segments
    text = " ".join(str(w["word"]).strip() for w in words)
    try:
        from mosestokenizer import MosesSentenceSplitter
        with MosesSentenceSplitter(opts.lan) as split:
            sents = split([text]) if text.strip() else []
    except ImportError:
        from sentence_splitter import SentenceSplitter
        sents = SentenceSplitter(language=opts.lan).split(text=text)
    out, i = [], 0
    for s in sents:
        n = len(s.split())
        if n:
            out.append(list(range(i, min(i + n, len(words)))))
            i += n
    if i < len(words):
        out.append(list(range(i, len(words))))
    return [g for g in out if g]


def _embedder(opts):
    """Encoding function; the model is loaded at most once per process."""
    key = (opts.embed_model, opts.embed_device, opts.embed_fp16, opts.embed_batch_size)
    if key not in _ENCODERS:
        _ENCODERS[key] = _build_embedder(opts)
    return _ENCODERS[key]


def _build_embedder(opts):
    import numpy as np
    if opts.embed_model == "hash":  # testing only: bag of character trigrams, NOT cross-lingual
        def enc(texts):
            m = np.zeros((len(texts), 4096), dtype=np.float32)
            for r, t in enumerate(texts):
                t = f"  {t.lower()}  "
                for q in range(len(t) - 2):
                    m[r, hash(t[q:q + 3]) % 4096] += 1
            return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        return enc
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(opts.embed_model, device=opts.embed_device)
    if opts.embed_fp16 and str(model.device).startswith("cuda"):
        model = model.half()
    model.eval()
    print(f"loaded {opts.embed_model} on {model.device} in {time.time() - t0:.1f}s", file=sys.stderr)

    def enc(texts):
        # float32 cast: with fp16 the raw output would break the plain dot products below
        return np.asarray(model.encode(texts, batch_size=opts.embed_batch_size,
                                       normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)
    return enc


def _run_vecalign(src_texts, tgt_texts, opts, enc):
    """Original Vecalign with our own embeddings instead of LASER files. Returns (units, src_emb):
    units = [(src_start, src_len, tgt_start, tgt_len)] in order, src_emb[i] = embedding of sentence i."""
    import numpy as np
    src = use_submodule("vecalign", opts.vecalign_dir)
    if src and not getattr(_run_vecalign, "_said", False):
        print(f"vecalign: using vecalign from {src}", file=sys.stderr)
        _run_vecalign._said = True
    try:
        from vecalign.dp_utils import (yield_overlaps, make_doc_embedding, make_alignment_types,
                                       preprocess_line, vecalign)
    except ImportError as e:
        sys.exit(f"vecalign not found or not built ({e}): run ./setup_aligners.sh")

    n = opts.max_size
    overlaps = sorted(set(yield_overlaps(src_texts, n)) | set(yield_overlaps(tgt_texts, n)))
    E = np.asarray(enc(overlaps), dtype=np.float32)
    sent2line = {t: k for k, t in enumerate(overlaps)}
    stack = vecalign(vecs0=make_doc_embedding(sent2line, E, src_texts, n),
                     vecs1=make_doc_embedding(sent2line, E, tgt_texts, n),
                     final_alignment_types=make_alignment_types(n),
                     del_percentile_frac=opts.del_percentile,
                     width_over2=math.ceil(n / 2.0) + opts.search_buffer_size,
                     max_size_full_dp=opts.max_size_full_dp,
                     costs_sample_size=20000, num_samps_for_norm=100)

    units, next_src, next_tgt = [], 0, 0
    for xs, ys in stack[0]["final_alignments"]:
        units.append((min(xs) if xs else next_src, len(xs), min(ys) if ys else next_tgt, len(ys)))
        next_src = max(xs) + 1 if xs else next_src
        next_tgt = max(ys) + 1 if ys else next_tgt
    src_emb = [E[sent2line[preprocess_line(t)[:10000]]] for t in src_texts]
    print(f"vecalign: {len(src_texts)} source vs {len(tgt_texts)} target sentences, {len(units)} units",
          file=sys.stderr)
    return units, src_emb


def _refine_splits(split_units, words, src_emb, src_texts, enc, opts):
    """For n:m units, choose where to cut the target words between the n source sentences: the cut
    points (whole words, within --split-window of the time/proportional guess) that maximise the sum of
    cos(source sentence, target piece), with a length prior and a bonus for cutting after punctuation."""
    W = opts.split_window
    plans, texts = [], {}
    for si, a, tw, prior in split_units:
        L = len(tw)
        allowed = [sorted({min(max(0, p + d), L) for d in range(-W, W + 1)}) for p in prior]
        starts, ends = [[0]] + allowed, allowed + [[L]]
        for q in range(a):
            for x in starts[q]:
                for y in ends[q]:
                    if x < y:
                        texts.setdefault(" ".join(str(words[tw[k]]["word"]) for k in range(x, y)), None)
        plans.append((si, a, tw, starts, ends))
    if not plans:
        return {}
    keys = list(texts)
    E = enc(keys)
    emb = {t: E[n] for n, t in enumerate(keys)}
    punct = (",", ".", ";", ":", "?", "!", "…", "–", "—")
    out = {}
    for si, a, tw, starts, ends in plans:
        def score(q, x, y):
            if x == y:  # empty piece: this source sentence gets no target words
                return 0.0
            t = " ".join(str(words[tw[k]]["word"]) for k in range(x, y))
            sc = float(src_emb[si + q] @ emb[t])
            # mild length prior: pieces much longer/shorter (in characters) than the source are suspicious
            sc -= opts.split_length_weight * abs(math.log((len(t) + 1) / (len(src_texts[si + q]) + 1)))
            if y < len(tw) and str(words[tw[y - 1]]["word"]).rstrip("\"'”»)").endswith(punct):
                sc += opts.split_punct_bonus
            return sc

        best = {0: (0.0, [])}  # DP over source sentences and cut positions
        for q in range(a):
            nxt = {}
            for x, (sc0, path) in best.items():
                for y in ends[q]:
                    if y < x:
                        continue
                    v = sc0 + score(q, x, y)
                    if y not in nxt or v > nxt[y][0]:
                        nxt[y] = (v, path + [y])
            best = nxt
        if len(tw) not in best:
            continue
        cuts, q = best[len(tw)][1], 0
        for n, k in enumerate(tw):
            while q < a - 1 and n >= cuts[q]:
                q += 1
            out[k] = si + q
    return out


def _split_unit(tw, si, a, words, src_texts, recs, sent_of, timed):
    """First guess of how to divide an n:m unit: by word timestamps, else proportionally to length."""
    if timed:
        ends = [r["end"] for r in recs[si:si + a]]
        q = 0
        for k in tw:
            mid = (words[k]["start"] + words[k]["end"]) / 2
            while q < a - 1 and mid > ends[q]:
                q += 1
            sent_of[k] = si + q
    else:
        lens = [max(1, len(src_texts[si + q].split())) for q in range(a)]
        bounds, cum = [], 0
        for L in lens:
            cum += L
            bounds.append(cum / sum(lens) * len(tw))
        q = 0
        for n, k in enumerate(tw):
            while q < a - 1 and n >= bounds[q]:
                q += 1
            sent_of[k] = si + q


def align_by_vecalign(words, segments, recs, ref_system, opts):
    tgt = _split_target_sentences(words, segments, opts)
    tgt_texts = [" ".join(str(words[k]["word"]) for k in g) for g in tgt]
    src_texts = [r["text"].get(ref_system, "") for r in recs]
    timed = has_times(words)

    enc = _embedder(opts)
    units, src_emb = _run_vecalign(src_texts, tgt_texts, opts, enc)

    sent_of = [None] * len(words)
    labels = [None] * len(recs)
    split_units, pending, last = [], [], 0
    for si, a, tj, b in units:
        tw = [k for g in tgt[tj:tj + b] for k in g]
        if a == 0:  # unaligned target sentence(s): attach to the previous source sentence
            if si > 0:
                for k in tw:
                    sent_of[k] = si - 1
                labels[si - 1] = (labels[si - 1] or "") + "+ins"
            else:
                pending += tw
            continue
        if pending:  # 0:1 before the first source sentence -> attach to it
            tw, pending = pending + tw, []
            labels[si] = "+ins"
        if b == 0:
            labels[si] = "vecalign 1:0" + (labels[si] or "")
            continue
        lab = f"vecalign {a}:{b}"
        if a == 1:
            for k in tw:
                sent_of[k] = si
        else:  # divide the target words among the a source sentences (never merge the sentences)
            lab += " split"
            _split_unit(tw, si, a, words, src_texts, recs, sent_of, timed)
            split_units.append((si, a, tw, [sum(1 for k in tw if sent_of[k] < si + q) for q in range(1, a)]))
        for q in range(a):
            labels[si + q] = lab + (labels[si + q] or "")
        last = si + a - 1
    for k in pending:
        sent_of[k] = last

    if split_units and opts.split_method == "embed":
        refined = _refine_splits(split_units, words, src_emb, src_texts, enc, opts)
        moved = sum(1 for k, v in refined.items() if sent_of[k] != v)
        sent_of = [refined.get(k, v) for k, v in enumerate(sent_of)]
        print(f"vecalign: {len(split_units)} n:m unit(s) split by embeddings ({moved} word(s) moved vs. "
              f"the {'time' if timed else 'proportional'} split)", file=sys.stderr)

    cur = 0
    for k in range(len(words)):
        if sent_of[k] is None:
            sent_of[k] = cur
        cur = sent_of[k]
    labels = [l or "vecalign 1:0" for l in labels]
    counts = Counter(l.split(" ")[1] if " " in l else l for l in labels)
    print(f"vecalign: {len(units)} units; per sentence: "
          + ", ".join(f"{k} x{v}" for k, v in counts.most_common()), file=sys.stderr)
    return sent_of, labels