#!/usr/bin/env python3
"""Convert Whisper-style diarized JSON (list of segments with words) to JSONL
with one segment per line.

Usage:
    python convert_to_jsonl.py input.json output.jsonl
    python convert_to_jsonl.py input.json output.jsonl --keep-speaker
"""
import argparse
import difflib
import json
import sys

from fix_punct import fix_punct


def fix_words(words):
    """Apply fix_punct to the word sequence and split the result back onto the words.

    Words keep a leading space only if they are space-separated from the previous
    word after the fix: [" Sechs", " -Minuten", " -Takt"] -> [" Sechs", "-Minuten", "-Takt"],
    [" d'", " Sitzung"] -> [" d'", "Sitzung"], [" 800", " 1000000"] -> [" 800", " Milliounen"].
    """
    raw = "".join(w["word"] for w in words)
    owner = [i for i, w in enumerate(words) for _ in w["word"]]  # word index of each raw char
    fixed = fix_punct(raw)
    if fixed == raw:
        return words

    pieces = [[] for _ in words]
    sm = difflib.SequenceMatcher(None, raw, fixed, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "delete":
            continue
        for k, j in enumerate(range(j1, j2)):
            if op == "insert":  # inserted text belongs to the word before it
                i = max(i1 - 1, 0)
            else:  # equal / replace: map position by position, clamped to the replaced span
                i = min(i1 + k, i2 - 1)
            pieces[owner[i]].append(fixed[j])

    out = []
    for w, chars in zip(words, pieces):
        text = "".join(chars)
        if text.strip():  # a word whose text vanished completely is dropped
            out.append({**w, "word": text})
    return out


def respace_from_text(words, text):
    """Give each word a leading space exactly when the segment text has a space before it.

    Whisper's word list can lose the left space where the text has one, e.g. at an internal
    sub-segment boundary: text "ass. D' Bänk", words [" ass.", "D'", " Bänk"]. The text is taken
    as the reference. A word is matched only right at the current position in the text (after
    optional whitespace); if the text and the words diverge, the word keeps its own spacing.
    """
    out, cur = [], 0
    for w in words:
        tok = w["word"].strip()
        idx = text.find(tok, cur)
        if idx < 0 or text[cur:idx].strip():
            out.append(w)  # text and words disagree here; keep the word as it is
            continue
        space = idx == 0 or text[idx - 1].isspace()
        out.append({**w, "word": (" " if space else "") + tok})
        cur = idx + len(tok)
    return out


def convert(segments, keep_speaker=False):
    for seg in segments:
        words = []
        for w in seg.get("words", []):
            text = w.get("word", "")
            if not text.strip() or w.get("start") is None or w.get("end") is None:
                continue
            words.append({"word": text, "start": w["start"], "end": w["end"]})
        if "text" in seg:
            words = respace_from_text(words, seg["text"])
        # a segment always starts a new token: make sure its first word carries the left space
        if words and not words[0]["word"][0].isspace():
            words[0]["word"] = " " + words[0]["word"]
        words = fix_words(words)

        start = seg.get("start", words[0]["start"] if words else None)
        end = seg.get("end", words[-1]["end"] if words else None)
        if start is None or end is None:
            continue

        out = {
            "segment": fix_punct(seg.get("text", "".join(w["word"] for w in words)).strip()),
            "start": start,
            "end": end,
            "words": words,
        }
        if "".join(w["word"] for w in words).strip() != out["segment"]:
            print(f"WARNING: words do not match the segment text at {start:.2f}s: "
                  f"{out['segment'][:60]!r}", file=sys.stderr)
        if keep_speaker and "speaker" in seg:
            out["speaker"] = seg["speaker"]
        yield out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input JSON file (list of segments)")
    ap.add_argument("output", help="output JSONL file")
    ap.add_argument("--keep-speaker", action="store_true",
                    help="add the segment's speaker label to each output line")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        segments = json.load(f)

    with open(args.output, "w", encoding="utf-8") as f:
        for line in convert(segments, args.keep_speaker):
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()