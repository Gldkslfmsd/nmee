#!/usr/bin/env python3
"""Build and extend a sentence-level, multi-system JSONL for one document.

Iteration 0 -- create: read the sentence segmentation made by asr_sentences.py
(one sentence per line: {"segment", "start", "end", "words"}) and turn it into the aligned format.
Its text becomes the first system (--name, e.g. canary_asr).

Iteration 1..n -- add: read the output of this script and add one more system (--add FILE), split into
the existing sentences by one of three methods (--align):

  mwer      Same-language systems (gold transcript vs. ASR, reference vs. MT, ASR vs. ASR).
            The added text is re-segmented to the sentences of --align-to (a system in the same
            language) with mweralign (https://github.com/mjpost/mweralign; Post & Hoang 2025,
            AS-WER of Matusov et al. 2005), using its SentencePiece tokenization (--mwer-tokenizer,
            default spm32k). Needs no timestamps. Used from the git submodule third_party/mweralign
            (see setup_aligners.sh), or from an installed mweralign package.
  vecalign  Other languages (translations). The added text is split into sentences (Moses), then
            aligned to the sentences of --align-to (default: the segmentation system) with the original
            Vecalign (Thompson & Koehn 2019; https://github.com/thompsonb/vecalign), called in-process
            with multilingual sentence embeddings (LaBSE by default, --embed-model) of all overlaps
            (concatenations of up to --max-size sentences). Vecalign itself only groups whole
            sentences; when one target sentence spans several source sentences, this script splits it
            (see "split" below). Vecalign is used from the git submodule third_party/vecalign (see
            setup_aligners.sh), or from an installed vecalign package. Also needs sentence-transformers,
            numpy, and mosestokenizer or sentence-splitter.
  time      Word timestamps only: every word goes to the sentence it overlaps most in time.

--add accepts Canary-style JSONL ("words" with "start"/"end" per line), word-per-line JSONL, a NeMo
hypothesis JSON, or plain text (e.g. a gold transcript or reference, one or more lines; no timestamps,
so not usable with --align time).

Output: one sentence per line:
  {"document", "dataset", "src_language",
   "systems_info": {"canary_asr": {"lan": "en", "is_human": false}, ...},
   "segmented_by": "canary_asr",
   "audio": "<audio-dir>/<doc>/<doc>.<NNNN>.wav",     # clip made by segment_audio.py
   "beg": 12.3, "end": 17.8,                            # seconds in the original long audio
   "text": {"canary_asr": "...", "canary_mt_cs": "..."},
   "alignment": {"canary_mt_cs": "vecalign 1:1", ...},  # how each added system was aligned here
   "words": {"canary_asr": [{"word", "start", "end"}, ...], ...}}   # only with --with-words

Alignment labels: "mwer", "time", or "vecalign a:b" where a source sentences were aligned to b target
sentences; "split" means the target sentence(s) of an n:m unit were divided among the n source sentences
(existing sentences are never merged): by default at the word boundaries where the pieces' embeddings
match the source sentences best (--split-method embed), or by time / proportionally (--split-method prior); "+ins" means unaligned target sentences (0:1) were
attached to this sentence; "1:0" means nothing was aligned to this sentence.

Usage:
    # 0) create from the sentence segmentation
    python add_and_align_sentences.py --input sentences/DOC.en.jsonl --output aligned/DOC.jsonl \
        --dataset earnings25 --src-language en --name canary_asr --lan en --audio-dir segments
    # 1..n) add one system per call; --output may be the same file as --input
    python add_and_align_sentences.py --input aligned/DOC.jsonl --output aligned/DOC.jsonl \
        --add outputs/DOC.cs.jsonl --name canary_mt_cs --lan cs --align vecalign
    python add_and_align_sentences.py --input aligned/DOC.jsonl --output aligned/DOC.jsonl \
        --add gold/DOC.txt --name gold_transcript --lan en --is-human --align mwer

--name, --lan and --is-human describe the system of the input sentences when creating, and the added
system when adding. Creating and adding are separate calls.

Metadata arguments (--document, --dataset, --src-language, --segmented-by, --audio-dir) are optional
when the input is already aligned: fields whose argument is given are updated, all others are kept as
they are in the input. When creating, fields whose argument is not given are set to null (only --name
is required). The document ID, if neither given nor already in the input, is taken from the --input
file name: DOC.jsonl, or DOC.<lan>.jsonl where <lan> is --lan or --src-language.
"""
import argparse
import json
import math
import sys
from pathlib import Path

# the aligners are git submodules under third_party/ (see setup_aligners.sh); an installed package of
# the same name is used only if the submodule is not there
THIRD_PARTY = [Path(__file__).resolve().parent / "third_party", Path("third_party")]


def use_submodule(name, extra_dir=None):
    """Put the submodule's copy of `name` first on sys.path, if it is present and built."""
    roots = ([Path(extra_dir)] if extra_dir else []) + [d / name for d in THIRD_PARTY]
    for root in roots:
        for path in (root, root / "python"):  # vecalign: repo root; mweralign: repo/python
            if (path / name / "__init__.py").exists():
                sys.path.insert(0, str(path))
                return str(path)
    return None


# ---------------------------------------------------------------- I/O

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_add(path):
    """Return (words, segments): words = [{"word", "start"?, "end"?}], segments = list of word-index lists
    (the file's own lines/segments). Plain text gives words without timestamps."""
    text = Path(path).read_text(encoding="utf-8")
    records = None
    try:
        data = json.loads(text)
        records = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        try:
            records = [json.loads(l) for l in text.splitlines() if l.strip()]
        except json.JSONDecodeError:
            records = None
    words, segments = [], []

    def add(ws):
        idx = []
        for w in ws:
            if str(w.get("word", "")).strip():
                idx.append(len(words))
                words.append(w)
        if idx:
            segments.append(idx)

    if records is None or not all(isinstance(r, dict) for r in records):  # plain text
        for line in text.splitlines():
            add([{"word": t} for t in line.split()])
        return words, segments
    for r in records:
        if "words" in r:
            add(r["words"])
        elif isinstance(r.get("timestamp"), dict) and "word" in r["timestamp"]:
            add(r["timestamp"]["word"])
        elif "word" in r:
            add([r])
        elif "segment" in r or "text" in r:
            add([{"word": t} for t in str(r.get("segment", r.get("text", ""))).split()])
    return words, segments


def has_times(words):
    return bool(words) and all("start" in w and "end" in w for w in words)


def is_aligned_format(recs):
    return bool(recs) and isinstance(recs[0].get("text"), dict) and "systems_info" in recs[0]


# ---------------------------------------------------------------- create

def create(segs, args):
    if args.name is None:
        sys.exit("--name is required when the input is asr_sentences.py output")
    document = args.document or doc_from_filename(args)
    unset = [a for a in ("dataset", "src_language", "lan") if getattr(args, a) is None]
    if unset:
        print(f"note: not given, set to null: {', '.join('--' + a.replace('_', '-') for a in unset)}",
              file=sys.stderr)
    name = args.name
    out = []
    for i, s in enumerate(segs):
        rec = {
            "document": document,
            "dataset": args.dataset,
            "src_language": args.src_language,
            "systems_info": {name: {"lan": args.lan, "is_human": args.is_human}},
            "segmented_by": args.segmented_by or name,
            "audio": None,
            "beg": s["start"],
            "end": s["end"],
            "text": {name: s.get("segment", "")},
        }
        if args.with_words:
            rec["words"] = {name: s.get("words", [])}
        out.append(rec)
    if args.audio_dir:
        set_audio(out, args.audio_dir)
    print(f"created {len(out)} sentences, system {name!r}", file=sys.stderr)
    return out


def doc_from_filename(args):
    """Document ID from the --input file name: DOC.jsonl or DOC.<lan>.jsonl."""
    name = Path(args.input).name
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else Path(name).stem
    for lan in (args.lan, args.src_language):
        if lan and stem.endswith(f".{lan}"):
            stem = stem[: -len(lan) - 1]
            break
    print(f"note: document ID taken from the input file name: {stem!r}", file=sys.stderr)
    return stem


def set_audio(recs, audio_dir):
    missing = 0
    for i, r in enumerate(recs):
        doc = r["document"]
        if not doc:
            sys.exit("--audio-dir needs the document ID (--document, or \"document\" in the input)")
        r["audio"] = str(Path(audio_dir) / doc / f"{doc}.{i:04d}.wav")
        missing += not Path(r["audio"]).exists()
    if missing:
        print(f"WARNING: {missing}/{len(recs)} audio clips not found under {audio_dir} "
              f"(run segment_audio.py on the same sentence files)", file=sys.stderr)


def update_metadata(recs, args):
    """Aligned input: update only the fields whose argument was given; keep all others."""
    changed = []
    if args.document is None and not recs[0].get("document"):
        doc = doc_from_filename(args)
        for r in recs:
            r["document"] = doc
        changed.append("document")
    for arg, field in (("document", "document"), ("dataset", "dataset"),
                       ("src_language", "src_language"), ("segmented_by", "segmented_by")):
        val = getattr(args, arg)
        if val is not None:
            for r in recs:
                r[field] = val
            changed.append(field)
    if args.audio_dir:
        set_audio(recs, args.audio_dir)
        changed.append("audio")
    if changed:
        print(f"updated: {', '.join(changed)}", file=sys.stderr)
    return recs


# ---------------------------------------------------------------- align: time

def assign_by_time(words, recs):
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
                       key=lambda q: 0 if begs[q] <= mid <= ends[q] else min(abs(mid - begs[q]), abs(mid - ends[q])))
            n_gap += 1
        best = max(best, j)
        j = best
        out.append(best)
    print(f"time: {n_gap} word(s) outside any sentence, assigned to the nearest", file=sys.stderr)
    return out, ["time"] * len(recs)


# ---------------------------------------------------------------- align: mwer (mweralign)

def assign_by_mwer(words, recs, ref_system, args):
    """Re-segment the added words to the sentences of ref_system with mweralign
    (https://github.com/mjpost/mweralign, AS-WER algorithm of mwerSegmenter), then map the output
    lines back to the added words."""
    src = use_submodule("mweralign", args.mweralign_dir)
    try:
        from mweralign import align_texts
    except ImportError:
        sys.exit("mweralign not found: run ./setup_aligners.sh (git submodule) or pip install mweralign")
    if src:
        print(f"mwer: using mweralign from {src}", file=sys.stderr)

    segmenter = None
    if args.mwer_tokenizer == "cj":
        from mweralign.segmenter import CJSegmenter
        segmenter = CJSegmenter()
    elif args.mwer_tokenizer not in ("none", "whitespace"):
        from mweralign import models
        from mweralign.segmenter import SPSegmenter
        segmenter = SPSegmenter(models.resolve(args.mwer_tokenizer))
    non_whitespace = args.lan in ("zh", "ja")
    from mweralign.segmenter import SPSegmenter as _SP
    is_tokenized = isinstance(segmenter, _SP) and not non_whitespace

    def tok(text):
        text = " ".join(text.split())
        return " ".join(segmenter.encode(text)) if segmenter is not None else text

    refs = []
    for r in recs:
        t = r["text"].get(ref_system, "").strip()
        refs.append(tok(t) if t else "_")  # an empty line would change the number of reference segments
    hyp = tok(" ".join(str(w["word"]).strip() for w in words))
    result = align_texts("\n".join(refs), hyp, is_tokenized=is_tokenized,
                         forbid_midword_boundary=is_tokenized)
    lines = result.split("\n")
    if len(lines) > len(recs) and not "".join(lines[len(recs):]).strip():
        lines = lines[:len(recs)]
    if len(lines) != len(recs):
        sys.exit(f"mweralign returned {len(lines)} lines for {len(recs)} sentences")
    if segmenter is not None:
        lines = [segmenter.decode(l) for l in lines]

    # map back by characters (ignoring whitespace): each word goes to the line of its first character
    line_of_char = [k for k, l in enumerate(lines) for c in l if not c.isspace()]
    sent_of, pos, mismatch = [], 0, 0
    for w in words:
        chars = [c for c in str(w["word"]) if not c.isspace()]
        k = line_of_char[pos] if pos < len(line_of_char) else len(recs) - 1
        sent_of.append(k)
        pos += len(chars)
    if len(line_of_char) != sum(len([c for c in str(w["word"]) if not c.isspace()]) for w in words):
        mismatch = 1
        print("WARNING: mweralign output differs from the input characters; the word mapping may be off "
              "(try --mwer-tokenizer none)", file=sys.stderr)
    cur = 0
    for i in range(len(sent_of)):  # keep monotonic
        sent_of[i] = cur = max(sent_of[i], cur)
    print(f"mwer: mweralign ({args.mwer_tokenizer}) re-segmented {len(words)} words into {len(recs)} sentences",
          file=sys.stderr)
    return sent_of, ["mwer"] * len(recs)


# ---------------------------------------------------------------- align: vecalign

def split_target_sentences(words, segments, args):
    """Target sentences as lists of word indices."""
    if args.tgt_split == "segments":
        return segments
    text = " ".join(str(w["word"]).strip() for w in words)
    try:
        from mosestokenizer import MosesSentenceSplitter
        with MosesSentenceSplitter(args.lan) as split:
            sents = split([text]) if text.strip() else []
    except ImportError:
        from sentence_splitter import SentenceSplitter
        sents = SentenceSplitter(language=args.lan).split(text=text)
    out, i = [], 0
    for s in sents:
        n = len(s.split())
        if n:
            out.append(list(range(i, min(i + n, len(words)))))
            i += n
    if i < len(words):
        out.append(list(range(i, len(words))))
    return [g for g in out if g]


def embedder(model_name):
    if model_name == "hash":  # testing only: bag of character trigrams, NOT cross-lingual
        import numpy as np

        def enc(texts):
            m = np.zeros((len(texts), 4096), dtype=np.float32)
            for r, t in enumerate(texts):
                t = f"  {t.lower()}  "
                for q in range(len(t) - 2):
                    m[r, hash(t[q:q + 3]) % 4096] += 1
            n = np.linalg.norm(m, axis=1, keepdims=True)
            return m / np.maximum(n, 1e-9)
        return enc
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def vecalign(src_texts, tgt_texts, args, enc):
    """Align with the original Vecalign (https://github.com/thompsonb/vecalign), using our own sentence
    embeddings (LaBSE by default) instead of LASER files. Returns (units, S): units =
    [(src_start, src_len, tgt_start, tgt_len)] in order, S = {(i, 1): embedding of source sentence i}."""
    import numpy as np
    from math import ceil
    src = use_submodule("vecalign", args.vecalign_dir)
    if src:
        print(f"vecalign: using vecalign from {src}", file=sys.stderr)
    try:
        from vecalign.dp_utils import (yield_overlaps, make_doc_embedding, make_alignment_types,
                                       preprocess_line, vecalign as run_vecalign)
    except ImportError as e:
        sys.exit(f"vecalign not found or not built ({e}): run ./setup_aligners.sh")
    n = args.max_size
    overlaps = sorted(set(yield_overlaps(src_texts, n)) | set(yield_overlaps(tgt_texts, n)))
    E = np.asarray(enc(overlaps), dtype=np.float32)
    sent2line = {t: k for k, t in enumerate(overlaps)}
    vecs0 = make_doc_embedding(sent2line, E, src_texts, n)
    vecs1 = make_doc_embedding(sent2line, E, tgt_texts, n)
    stack = run_vecalign(vecs0=vecs0, vecs1=vecs1,
                         final_alignment_types=make_alignment_types(n),
                         del_percentile_frac=args.del_percentile,
                         width_over2=ceil(n / 2.0) + args.search_buffer_size,
                         max_size_full_dp=args.max_size_full_dp,
                         costs_sample_size=20000,
                         num_samps_for_norm=100)
    alignments = stack[0]["final_alignments"]

    units, next_src, next_tgt = [], 0, 0
    for xs, ys in alignments:
        si = min(xs) if xs else next_src
        tj = min(ys) if ys else next_tgt
        units.append((si, len(xs), tj, len(ys)))
        next_src = max(xs) + 1 if xs else next_src
        next_tgt = max(ys) + 1 if ys else next_tgt
    S = {(i, 1): E[sent2line[preprocess_line(t)[:10000]]] for i, t in enumerate(src_texts)}
    print(f"vecalign: {len(src_texts)} source vs {len(tgt_texts)} target sentences, {len(units)} units",
          file=sys.stderr)
    return units, S


def refine_splits(split_units, words, src_emb, src_texts, enc, args):
    """For n:m units, choose where to cut the target words between the n source sentences: the cut
    points (whole words, within --split-window words of the time/proportional guess) that maximise the
    sum of cos(source sentence, target piece) + a small bonus for cutting after punctuation."""
    import numpy as np
    W = args.split_window
    plans, texts = [], {}
    for si, a, tw, prior in split_units:
        L = len(tw)
        allowed = [sorted({min(max(0, p + d), L) for d in range(-W, W + 1)}) for p in prior]
        starts = [[0]] + allowed
        ends = allowed + [[L]]
        spans = {(x, y) for q in range(a) for x in starts[q] for y in ends[q] if x < y}
        for x, y in spans:
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
            sc -= args.split_length_weight * abs(math.log((len(t) + 1) / (len(src_texts[si + q]) + 1)))
            if y < len(tw) and str(words[tw[y - 1]]["word"]).rstrip("\"'”»)").endswith(punct):
                sc += args.split_punct_bonus
            return sc
        # DP over source sentences q and cut position x (start of piece q)
        best = {0: (0.0, [])}
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
        cuts = best[len(tw)][1]
        q = 0
        for n, k in enumerate(tw):
            while q < a - 1 and n >= cuts[q]:
                q += 1
            out[k] = si + q
    return out


def assign_by_vecalign(words, segments, recs, ref_system, args):
    import numpy as np
    tgt = split_target_sentences(words, segments, args)
    tgt_texts = [" ".join(str(words[k]["word"]) for k in g) for g in tgt]
    src_texts = [r["text"].get(ref_system, "") for r in recs]
    N, M = len(src_texts), len(tgt_texts)
    timed = has_times(words)

    enc = embedder(args.embed_model)
    units, S = vecalign(src_texts, tgt_texts, args, enc)
    split_units = []

    sent_of = [None] * len(words)
    labels = [None] * len(recs)
    pending = []  # target words of 0:1 units waiting for a sentence
    last = 0
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
            tw = pending + tw
            pending = []
            labels[si] = "+ins"
        if b == 0:
            labels[si] = f"vecalign 1:0" + (labels[si] or "")
            continue
        lab = f"vecalign {a}:{b}"
        if a == 1:
            for k in tw:
                sent_of[k] = si
        else:  # divide the target words among the a source sentences
            lab += " split"
            if timed:
                ends_ = [r["end"] for r in recs[si:si + a]]
                q = 0
                for k in tw:
                    mid = (words[k]["start"] + words[k]["end"]) / 2
                    while q < a - 1 and mid > ends_[q]:
                        q += 1
                    sent_of[k] = si + q
            else:
                lens = [max(1, len(src_texts[si + q].split())) for q in range(a)]
                tot = sum(lens)
                cum, bounds = 0, []
                for L in lens:
                    cum += L
                    bounds.append(cum / tot * len(tw))
                q = 0
                for n, k in enumerate(tw):
                    while q < a - 1 and n >= bounds[q]:
                        q += 1
                    sent_of[k] = si + q
        if a > 1:
            prior = [sum(1 for k in tw if sent_of[k] < si + q) for q in range(1, a)]
            split_units.append((si, a, tw, prior))
        for q in range(a):
            labels[si + q] = lab + (labels[si + q] or "")
        last = si + a - 1
    for k in pending:
        sent_of[k] = last
    if split_units and args.split_method == "embed":
        src_emb = [S[(i, 1)] for i in range(N)]
        refined = refine_splits(split_units, words, src_emb, src_texts, enc, args)
        moved = sum(1 for k, v in refined.items() if sent_of[k] != v)
        for k, v in refined.items():
            sent_of[k] = v
        print(f"vecalign: {len(split_units)} n:m unit(s) split by embeddings "
              f"({moved} word(s) moved vs. the {'time' if timed else 'proportional'} split)", file=sys.stderr)
    cur = 0
    for k in range(len(words)):
        if sent_of[k] is None:
            sent_of[k] = cur
        cur = sent_of[k]
    labels = [l or "vecalign 1:0" for l in labels]
    from collections import Counter
    c = Counter(l.split(" ")[1] if " " in l else l for l in labels)
    print(f"vecalign: {len(units)} units; per sentence: " + ", ".join(f"{k} x{v}" for k, v in c.most_common()),
          file=sys.stderr)
    return sent_of, labels


# ---------------------------------------------------------------- add a system

def add_system(recs, words, segments, args):
    name = args.name
    info = recs[0]["systems_info"]
    if name in info and not args.overwrite:
        sys.exit(f"system {name!r} is already in the input (use --overwrite)")
    if args.lan is None:
        sys.exit("--lan is required with --add")
    if not words:
        sys.exit(f"{args.add}: no words")

    if args.align == "time":
        if not has_times(words):
            sys.exit("--align time needs word timestamps; use --align mwer or vecalign")
        sent_of, labels = assign_by_time(words, recs)
    elif args.align == "mwer":
        ref = args.align_to or next((s for s, v in info.items() if v.get("lan") == args.lan and s != name), None)
        if ref is None:
            sys.exit(f"--align mwer needs a system in the same language ({args.lan}) to align to; "
                     f"give --align-to (systems: {', '.join(info)})")
        if info.get(ref, {}).get("lan") not in (None, args.lan):
            print(f"WARNING: aligning {args.lan} to {ref!r} ({info[ref]['lan']}) with mwer; "
                  f"use vecalign for different languages", file=sys.stderr)
        print(f"mwer: aligning to {ref!r}", file=sys.stderr)
        sent_of, labels = assign_by_mwer(words, recs, ref, args)
    else:
        ref = args.align_to or recs[0]["segmented_by"]
        print(f"vecalign: aligning to {ref!r}", file=sys.stderr)
        sent_of, labels = assign_by_vecalign(words, segments, recs, ref, args)

    per_sent = [[] for _ in recs]
    for w, k in zip(words, sent_of):
        per_sent[k].append(w)
    for r, ws, lab in zip(recs, per_sent, labels):
        r["systems_info"][name] = {"lan": args.lan, "is_human": args.is_human}
        r["text"][name] = args.joiner.join(str(w["word"]).strip() for w in ws)
        r.setdefault("alignment", {})[name] = lab
        if args.with_words:
            r.setdefault("words", {})[name] = ws
        elif name in r.get("words", {}):
            del r["words"][name]  # overwritten system: don't keep its old words
    empty = sum(1 for ws in per_sent if not ws)
    print(f"added {name!r}: {len(words)} words -> {len(recs)} sentences, {empty} sentence(s) without words",
          file=sys.stderr)
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="asr_sentences.py output, or output of this script")
    ap.add_argument("--output", required=True, help="output JSONL (may be the same file as --input)")
    # metadata
    ap.add_argument("--document", help="document ID, e.g. the audio file stem")
    ap.add_argument("--dataset")
    ap.add_argument("--src-language")
    ap.add_argument("--segmented-by", help="name of the segmentation (default when creating: --name)")
    ap.add_argument("--audio-dir", help="OUT_DIR of segment_audio.py run on the same sentence files")
    # the system: of the input sentences when creating, of --add when adding
    ap.add_argument("--name", help="system name, e.g. canary_asr or canary_mt_cs")
    ap.add_argument("--lan", help="language of the system's text")
    ap.add_argument("--is-human", action="store_true", help="the system is a human (e.g. gold transcript)")
    # adding
    ap.add_argument("--add", help="file of the system to add (Canary JSONL with words, NeMo JSON, or plain text)")
    ap.add_argument("--align", choices=["mwer", "vecalign", "time"], default="vecalign")
    ap.add_argument("--align-to", help="system to align to (mwer: a system in the same language, default the "
                                       "first one with --lan; vecalign: default the segmentation system)")
    ap.add_argument("--mwer-tokenizer", default="spm32k",
                    help="mwer: mweralign tokenizer: spm32k (default, recommended by mweralign), spm64k, "
                         "spm128k, spm256k, a SentencePiece .model path, 'cj', or 'none' (whitespace)")
    ap.add_argument("--tgt-split", choices=["moses", "segments"], default="moses",
                    help="vecalign: split the added text into sentences with Moses (default), or use its own "
                         "lines/segments as sentences")
    ap.add_argument("--embed-model", default="sentence-transformers/LaBSE",
                    help="vecalign: sentence-transformers model (default LaBSE; 'hash' = testing only)")
    ap.add_argument("--max-size", type=int, default=4,
                    help="vecalign: Vecalign's --alignment_max_size: max sentences in one unit, e.g. 4 allows 1:3, 2:2, "
                         "3:1 (default 4)")
    ap.add_argument("--vecalign-dir", help="path to a vecalign checkout (default: third_party/vecalign)")
    ap.add_argument("--mweralign-dir", help="path to a mweralign checkout (default: third_party/mweralign)")
    ap.add_argument("--search-buffer-size", type=int, default=5,
                    help="vecalign: Vecalign's --search_buffer_size (default 5)")
    ap.add_argument("--max-size-full-dp", type=int, default=300,
                    help="vecalign: Vecalign's --max_size_full_dp (default 300)")
    ap.add_argument("--del-percentile", type=float, default=0.2,
                    help="vecalign: Vecalign's --del_percentile_frac (default 0.2)")
    ap.add_argument("--split-method", choices=["embed", "prior"], default="embed",
                    help="vecalign: how to divide one target sentence among several source sentences (n:m units): "
                         "embed = cut where the pieces match the source sentences best (default); prior = by "
                         "word timestamps, or proportionally to length without timestamps")
    ap.add_argument("--split-window", type=int, default=20,
                    help="vecalign: search cut points within this many words of the time/proportional guess")
    ap.add_argument("--split-length-weight", type=float, default=0.1,
                    help="vecalign: weight of the length prior when splitting (default 0.1)")
    ap.add_argument("--split-punct-bonus", type=float, default=0.03,
                    help="vecalign: score bonus for cutting after punctuation (default 0.03)")
    ap.add_argument("--joiner", default=" ", help="string between words (default: space; '' for zh/ja)")
    ap.add_argument("--overwrite", action="store_true", help="replace a system that is already present")
    ap.add_argument("--with-words", action="store_true", help="store words (with timestamps if any) per system")
    args = ap.parse_args()

    recs = read_jsonl(args.input)
    if not recs:
        sys.exit(f"{args.input} is empty")
    aligned = is_aligned_format(recs)
    if args.add and not aligned:
        sys.exit("create the aligned file first (without --add), then add systems one per call")
    recs = update_metadata(recs, args) if aligned else create(recs, args)
    if args.add:
        if not args.name:
            sys.exit("--name is required with --add")
        words, segments = load_add(args.add)
        recs = add_system(recs, words, segments, args)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.output) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as o:
        for r in recs:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    print(f"-> {args.output}: {len(recs)} sentences, systems: {', '.join(recs[0]['systems_info'])}",
          file=sys.stderr)


if __name__ == "__main__":
    main()