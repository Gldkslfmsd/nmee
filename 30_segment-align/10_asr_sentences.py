#!/usr/bin/env python3
"""Re-segment Canary output (with word timestamps) into sentences using the Moses sentence splitter.

Input: one JSONL file per document (e.g. <doc>.en.jsonl), one Canary segment per line:
    {"segment": "...", "start": 0.24, "end": 8.0, "start_offset": 3, "end_offset": 100,
     "words": [{"word": "which", "start": 0.24, "end": 0.32, "start_offset": 3, "end_offset": 4}, ...]}

Steps, per document:
  1. Join the words of all Canary segments into one text (Canary's own segment boundaries are ignored:
     NeMo splits at every word ending in . ? !, including "Mr." or "No.").
  2. Split the text into sentences with the Moses sentence splitter: by default the original
     split-sentences.perl via the `mosestokenizer` package (needs perl on PATH); --backend python uses
     `sentence-splitter`, a pure-Python port with the same per-language non-breaking prefixes.
     Each sentence gets the words it contains: start = first word's start, end = last word's end.
  3. Attach halves of the gaps: the pause between two consecutive sentences is split in the middle,
     so the sentences become contiguous (end of one = start of the next). The first sentence keeps its
     first word's start and the last one its last word's end.
  4. Minimum length: a sentence shorter than --min-dur seconds is merged into the shorter of its
     adjacent sentences; repeated (shortest first) until all are long enough or one sentence is left.

Output: OUT_DIR/<doc><suffix>, same format as the input (segment, start, end, start_offset, end_offset,
words), so segment_audio.py can read it. start_offset/end_offset are those of the first/last word.

Usage:
    pip install mosestokenizer          # or: pip install sentence-splitter  (with --backend python)
    python asr_sentences.py CANARY_DIR OUT_DIR --suffix .en.jsonl --lang en --min-dur 1.0
"""
import argparse
import json
import sys
from pathlib import Path


def load_words(path):
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words += [w for w in json.loads(line).get("words", []) if str(w.get("word", "")).strip()]
    return words


def make_splitter(backend, lang):
    """Return (split function: text -> list of sentences, close function)."""
    if backend == "perl":
        from mosestokenizer import MosesSentenceSplitter
        s = MosesSentenceSplitter(lang)
        # the whole document is passed as one line, so the splitter's line unwrapping plays no role
        return (lambda text: s([text]) if text.strip() else []), s.close
    from sentence_splitter import SentenceSplitter
    s = SentenceSplitter(language=lang)
    return (lambda text: s.split(text=text)), (lambda: None)


def split_sentences(words, split):
    """Group word indices into sentences. Moses only splits between tokens, so each sentence's
    whitespace tokens correspond 1:1 to consecutive ASR words."""
    text = " ".join(w["word"].strip() for w in words)
    groups, i = [], 0
    for sent in split(text):
        n = len(sent.split())
        if n == 0:
            continue
        g = list(range(i, min(i + n, len(words))))
        if " ".join(words[k]["word"].strip() for k in g) != " ".join(sent.split()):
            print(f"WARNING: sentence does not match the ASR words: {sent[:60]!r}", file=sys.stderr)
        if g:
            groups.append(g)
        i += n
    if i < len(words):
        groups.append(list(range(i, len(words))))
    return groups


def make_segments(groups, words):
    segs = []
    for g in groups:
        segs.append({"words": [words[k] for k in g],
                     "start": words[g[0]]["start"], "end": words[g[-1]]["end"]})
    # attach halves of the gaps between consecutive sentences
    for a, b in zip(segs, segs[1:]):
        mid = (a["end"] + b["start"]) / 2
        a["end"] = b["start"] = mid
    return segs


def merge_short(segs, min_dur):
    """Merge segments shorter than min_dur into their shorter neighbour, shortest first."""
    if min_dur <= 0:
        return segs
    segs = list(segs)
    while len(segs) > 1:
        durs = [s["end"] - s["start"] for s in segs]
        i = min(range(len(segs)), key=lambda k: durs[k])
        if durs[i] >= min_dur:
            break
        if i == 0:
            j = 1
        elif i == len(segs) - 1:
            j = i - 1
        else:
            j = i - 1 if durs[i - 1] <= durs[i + 1] else i + 1
        a, b = (segs[j], segs[i]) if j < i else (segs[i], segs[j])
        merged = {"words": a["words"] + b["words"], "start": a["start"], "end": b["end"]}
        k = min(i, j)
        segs[k:k + 2] = [merged]
    return segs


def to_record(seg):
    w = seg["words"]
    rec = {"segment": " ".join(x["word"].strip() for x in w),
           "start": round(seg["start"], 3), "end": round(seg["end"], 3)}
    if "start_offset" in w[0] and "end_offset" in w[-1]:
        rec["start_offset"], rec["end_offset"] = w[0]["start_offset"], w[-1]["end_offset"]
    rec["words"] = w
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("canary_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--suffix", default=".en.jsonl", help="input (and output) file suffix (default .en.jsonl)")
    ap.add_argument("--lang", default="en", help="language for the Moses non-breaking prefixes (default en)")
    ap.add_argument("--backend", choices=["perl", "python"], default="perl",
                    help="perl = original split-sentences.perl via mosestokenizer (default); "
                         "python = sentence-splitter port")
    ap.add_argument("--min-dur", type=float, default=1.0,
                    help="merge sentences shorter than this (seconds) into the shorter neighbour (default 1 = 1 second)")
    ap.add_argument("--how-many", type=int, default=None, help="process only the first N documents")
    args = ap.parse_args()

    split, close = make_splitter(args.backend, args.lang)

    files = sorted(Path(args.canary_dir).glob(f"*{args.suffix}"))
    if not files:
        sys.exit(f"no *{args.suffix} files in {args.canary_dir}")
    if args.how_many is not None:
        files = files[: args.how_many]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        stem = f.name[: -len(args.suffix)]
        words = load_words(f)
        if not words:
            print(f"WARNING: {f.name}: no words with timestamps", file=sys.stderr)
            continue
        back = sum(1 for a, b in zip(words, words[1:]) if b["start"] < a["start"] - 0.01)
        if back:
            print(f"WARNING: {f.name}: {back} word(s) start before the previous word", file=sys.stderr)
        n_canary = sum(1 for l in open(f, encoding="utf-8") if l.strip())

        groups = split_sentences(words, split)
        segs = merge_short(make_segments(groups, words), args.min_dur)
        with open(out_dir / f.name, "w", encoding="utf-8") as o:
            for s in segs:
                o.write(json.dumps(to_record(s), ensure_ascii=False) + "\n")
        durs = [s["end"] - s["start"] for s in segs]
        print(f"{stem}: {len(words)} words, {n_canary} Canary segments -> {len(groups)} sentences "
              f"-> {len(segs)} segments (min {min(durs):.2f}s, max {max(durs):.2f}s)")
    close()


if __name__ == "__main__":
    main()
