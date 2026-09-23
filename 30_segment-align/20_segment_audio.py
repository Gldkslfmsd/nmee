#!/usr/bin/env python3
"""Cut long-form audio into per-segment clips using timestamps from translation jsonl files.

For every <doc>.en.jsonl in --translations-dir (one JSON object per line with "segment" text and
"start"/"end" in seconds), reads <doc>.wav (or .mp3, .flac, ... see --audio-ext) from --audio-dir and writes
    OUT_DIR/<doc>/<doc>.<NNNN>.wav      (NNNN = 0-based line index in the jsonl)

Usage:
    python segment_audio.py OUT_DIR \
        --audio-dir ~/work/uedin/mtm26/nmee/iwslt26-cs-dev/audio \
        --translations-dir ~/work/uedin/mtm26/nmee/outputs/iwslt26-cs-dev \
        [--suffix .en.jsonl] [--pad 0.2] [--how-many 3] [--debug-prints DEBUG_DIR]

--how-many N          process only the first N documents (sorted by name), for debugging
--debug-prints DIR    write DIR/<doc>.txt with "start<TAB>end<TAB>text" per segment, plus DIR/stats.txt
                      with statistics (durations, shortest segments, empty texts, overlaps, segments
                      outside the audio); the statistics are also printed to stderr

Clips are always written as WAV. PCM WAV input is cut with the Python standard library (wave); other
formats (mp3, flac, ...) and WAVs that `wave` can't read are cut with `ffmpeg`, or `sox` if ffmpeg is
not on PATH (note that sox often lacks mp3 support).
"""
import argparse
import json
import shutil
import statistics
import subprocess
import sys
import wave
from pathlib import Path

SHORT_SEC = 0.5      # segments shorter than this are reported as "short"
N_SHORTEST = 10      # how many shortest segments to list


def read_segments(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_audio(audio_dir, stem, exts):
    for ext in exts:
        p = audio_dir / f"{stem}.{ext}"
        if p.exists():
            return p
    return None


def audio_duration(path):
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                return w.getnframes() / w.getframerate()
        except wave.Error:
            pass
    for cmd in (["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                ["soxi", "-D", str(path)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(out.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
    return None


def cut_with_wave(wav_path, segs, out_dir, stem, pad):
    with wave.open(str(wav_path), "rb") as w:
        params = w.getparams()
        sr, n = w.getframerate(), w.getnframes()
        for i, s in enumerate(segs):
            a = max(0, int((s["start"] - pad) * sr))
            b = min(n, int((s["end"] + pad) * sr))
            w.setpos(min(a, n))
            frames = w.readframes(max(0, b - a))
            with wave.open(str(out_dir / f"{stem}.{i:04d}.wav"), "wb") as o:
                o.setparams(params)
                o.writeframes(frames)


def cut_with_ffmpeg(path, segs, out_dir, stem, pad):
    for i, s in enumerate(segs):
        a = max(0.0, s["start"] - pad)
        dur = max(0.0, (s["end"] + pad) - a)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
                        str(out_dir / f"{stem}.{i:04d}.wav")], check=True)


def cut(path, segs, out_dir, stem, pad):
    if path.suffix.lower() == ".wav":
        try:
            return cut_with_wave(path, segs, out_dir, stem, pad)
        except wave.Error as e:
            print(f"{path}: wave module failed ({e}), using an external tool", file=sys.stderr)
    if shutil.which("ffmpeg"):
        cut_with_ffmpeg(path, segs, out_dir, stem, pad)
    elif shutil.which("sox"):
        cut_with_sox(path, segs, out_dir, stem, pad)
    else:
        sys.exit(f"{path}: needs ffmpeg or sox on PATH")


def cut_with_sox(wav_path, segs, out_dir, stem, pad):
    for i, s in enumerate(segs):
        a = max(0.0, s["start"] - pad)
        dur = max(0.0, (s["end"] + pad) - a)
        subprocess.run(["sox", str(wav_path), str(out_dir / f"{stem}.{i:04d}.wav"),
                        "trim", f"{a:.3f}", f"{dur:.3f}"], check=True)


# ---------------------------------------------------------------- debug output

def write_tsv(path, segs):
    with open(path, "w", encoding="utf-8") as f:
        for s in segs:
            text = " ".join(str(s.get("segment", "")).split())  # no tabs/newlines inside the text
            f.write(f"{s['start']:.2f}\t{s['end']:.2f}\t{text}\n")


def doc_problems(stem, segs, audio_len):
    """Return a list of (kind, message) for one document."""
    probs = []
    for i, s in enumerate(segs):
        where = f"{stem}#{i:04d} [{s['start']:.2f}-{s['end']:.2f}]"
        text = str(s.get("segment", "")).strip()
        dur = s["end"] - s["start"]
        if not text:
            probs.append(("empty text", where))
        if dur <= 0:
            probs.append(("non-positive duration", f"{where} {dur:.2f}s"))
        elif dur < SHORT_SEC:
            probs.append((f"shorter than {SHORT_SEC}s", f"{where} {dur:.2f}s {text[:60]!r}"))
        if i > 0 and s["start"] < segs[i - 1]["end"] - 1e-6:
            probs.append(("overlaps previous segment", f"{where} prev ends {segs[i - 1]['end']:.2f}"))
        if audio_len is not None and s["end"] > audio_len + 0.05:
            probs.append(("beyond end of audio", f"{where} audio is {audio_len:.2f}s"))
    return probs


def fmt_stats(durs):
    if not durs:
        return "no segments"
    return (f"n={len(durs)} total={sum(durs):.1f}s min={min(durs):.2f}s "
            f"median={statistics.median(durs):.2f}s mean={statistics.mean(durs):.2f}s max={max(durs):.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--translations-dir", required=True)
    ap.add_argument("--suffix", default=".en.jsonl", help="translation file suffix (default: .en.jsonl)")
    ap.add_argument("--audio-ext", default="wav,mp3,flac,ogg,opus,m4a",
                    help="audio extensions to look for, in order of preference (default: wav,mp3,flac,ogg,opus,m4a)")
    ap.add_argument("--pad", type=float, default=0.0, help="seconds of padding on each side of a clip")
    ap.add_argument("--how-many", type=int, default=None, help="how many documents to process, for debugging")
    ap.add_argument("--debug-prints", metavar="DIR", default=None,
                    help="write start/end/text TSV per document and statistics into DIR. It is named *.txt so that it can be open in Audacity.")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    exts = [e.strip().lstrip(".") for e in args.audio_ext.split(",") if e.strip()]
    debug_dir = Path(args.debug_prints) if args.debug_prints else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.translations_dir).glob(f"*{args.suffix}"))
    if not files:
        sys.exit(f"No *{args.suffix} files in {args.translations_dir}")
    if args.how_many is not None:
        files = files[: args.how_many]

    all_durs, all_segs, all_probs, doc_lines = [], [], [], []
    for jf in files:
        stem = jf.name[: -len(args.suffix)]
        wav = find_audio(audio_dir, stem, exts)
        segs = read_segments(jf)

        if debug_dir:
            write_tsv(debug_dir / f"{stem}.txt", segs)
            audio_len = audio_duration(wav) if wav else None
            durs = [s["end"] - s["start"] for s in segs]
            all_durs += durs
            all_segs += [(s["end"] - s["start"], stem, i, s) for i, s in enumerate(segs)]
            probs = doc_problems(stem, segs, audio_len)
            all_probs += probs
            alen = f"{audio_len:.1f}s" if audio_len is not None else "missing"
            covered = f" ({sum(d for d in durs if d > 0) / audio_len:.0%} covered)" if audio_len else ""
            doc_lines.append(f"{stem}: audio {alen}{covered}; {fmt_stats(durs)}; {len(probs)} issue(s)")

        if wav is None:
            print(f"WARNING: no audio for {stem} in {audio_dir} ({', '.join(exts)})", file=sys.stderr)
            continue
        out_dir = out_root / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cut(wav, segs, out_dir, stem, args.pad)
        print(f"{stem}: {len(segs)} clips -> {out_dir}")

    if debug_dir:
        lines = [f"documents: {len(files)}", f"all segments: {fmt_stats(all_durs)}", ""]
        lines.append(f"{N_SHORTEST} shortest segments:")
        for dur, stem, i, s in sorted(all_segs, key=lambda x: x[0])[:N_SHORTEST]:
            lines.append(f"  {dur:6.2f}s  {stem}#{i:04d} [{s['start']:.2f}-{s['end']:.2f}]  "
                         f"{str(s.get('segment', ''))[:70]!r}")
        lines.append("")
        kinds = {}
        for kind, msg in all_probs:
            kinds.setdefault(kind, []).append(msg)
        if kinds:
            lines.append("issues:")
            for kind, msgs in kinds.items():
                lines.append(f"  {kind}: {len(msgs)}")
                lines += [f"    {m}" for m in msgs]
        else:
            lines.append("issues: none (no empty texts, overlaps, non-positive or out-of-audio segments)")
        lines += ["", "per document:"] + [f"  {l}" for l in doc_lines]
        report = "\n".join(lines) + "\n"
        (debug_dir / "stats.txt").write_text(report, encoding="utf-8")
        print(report, file=sys.stderr)
        print(f"debug output: {debug_dir}/<doc>.txt, {debug_dir}/stats.txt", file=sys.stderr)


if __name__ == "__main__":
    main()