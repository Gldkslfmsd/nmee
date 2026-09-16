#!/usr/bin/env python3
"""Cut long-form audio into per-segment clips using timestamps from translation jsonl files.

For every <doc>.en.jsonl in --translations-dir (one JSON object per line with "start"/"end" in seconds),
reads <doc>.wav from --audio-dir and writes
    OUT_DIR/<doc>/<doc>.<NNNN>.wav      (NNNN = 0-based line index in the jsonl)

Usage:
    python segment_audio.py OUT_DIR \
        --audio-dir ~/work/uedin/mtm26/nmee/iwslt26-cs-dev/audio \
        --translations-dir ~/work/uedin/mtm26/nmee/outputs/iwslt26-cs-dev \
        [--suffix .en.jsonl] [--pad 0.2] [--only-segments annotations.jsonl]

Uses only the Python standard library (wave) for PCM WAV. If a file cannot be read by `wave`
(e.g. float WAV), it falls back to `sox` (must be on PATH).
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path


def read_segments(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cut_with_wave(wav_path, segs, out_dir, stem, pad, wanted):
    with wave.open(str(wav_path), "rb") as w:
        params = w.getparams()
        sr, n = w.getframerate(), w.getnframes()
        for i, s in enumerate(segs):
            if wanted is not None and i not in wanted:
                continue
            a = max(0, int((s["start"] - pad) * sr))
            b = min(n, int((s["end"] + pad) * sr))
            w.setpos(a)
            frames = w.readframes(max(0, b - a))
            with wave.open(str(out_dir / f"{stem}.{i:04d}.wav"), "wb") as o:
                o.setparams(params)
                o.writeframes(frames)


def cut_with_sox(wav_path, segs, out_dir, stem, pad, wanted):
    for i, s in enumerate(segs):
        if wanted is not None and i not in wanted:
            continue
        a = max(0.0, s["start"] - pad)
        dur = (s["end"] + pad) - a
        subprocess.run(["sox", str(wav_path), str(out_dir / f"{stem}.{i:04d}.wav"),
                        "trim", f"{a:.3f}", f"{dur:.3f}"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--translations-dir", required=True)
    ap.add_argument("--suffix", default=".en.jsonl", help="translation file suffix (default: .en.jsonl)")
    ap.add_argument("--pad", type=float, default=0.0, help="seconds of padding on each side of a clip")
    ap.add_argument("--only-segments", help="annotations jsonl with filename/segment; cut only those clips")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)

    wanted_by_stem = None
    if args.only_segments:
        wanted_by_stem = {}
        for l in open(args.only_segments, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("filename") is not None:
                    wanted_by_stem.setdefault(Path(r["filename"]).stem, set()).add(r["segment"])

    files = sorted(Path(args.translations_dir).glob(f"*{args.suffix}"))
    if not files:
        sys.exit(f"No *{args.suffix} files in {args.translations_dir}")

    for jf in files:
        stem = jf.name[: -len(args.suffix)]
        wanted = None if wanted_by_stem is None else wanted_by_stem.get(stem)
        if wanted_by_stem is not None and not wanted:
            continue
        wav = audio_dir / f"{stem}.wav"
        if not wav.exists():
            print(f"WARNING: missing audio {wav}", file=sys.stderr)
            continue
        segs = read_segments(jf)
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            cut_with_wave(wav, segs, out_dir, stem, args.pad, wanted)
        except wave.Error as e:
            print(f"{wav}: wave module failed ({e}), using sox", file=sys.stderr)
            cut_with_sox(wav, segs, out_dir, stem, args.pad, wanted)
        n = len(segs) if wanted is None else len(wanted)
        print(f"{stem}: {n} clips -> {out_dir}")


if __name__ == "__main__":
    main()
