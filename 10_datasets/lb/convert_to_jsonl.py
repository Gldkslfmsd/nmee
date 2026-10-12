#!/usr/bin/env python3
"""Convert Whisper-style diarized JSON (list of segments with words) to JSONL
with one segment per line.

Usage:
    python convert_to_jsonl.py input.json output.jsonl
    python convert_to_jsonl.py input.json output.jsonl --keep-speaker
"""
import argparse
import json

from fix_punct import fix_punct


def convert(segments, keep_speaker=False):
    for seg in segments:
        words = []
        for w in seg.get("words", []):
            text = w.get("word", "").strip()
            if not text or w.get("start") is None or w.get("end") is None:
                continue
            words.append({"word": text, "start": w["start"], "end": w["end"]})

        start = seg.get("start", words[0]["start"] if words else None)
        end = seg.get("end", words[-1]["end"] if words else None)
        if start is None or end is None:
            continue

        out = {
            "segment": fix_punct(seg.get("text", " ".join(w["word"] for w in words)).strip()),
            "start": start,
            "end": end,
            "words": words,
        }
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