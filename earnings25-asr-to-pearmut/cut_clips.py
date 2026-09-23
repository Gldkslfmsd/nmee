#!/usr/bin/env python3
"""Cut one audio clip per annotated Canary segment out of the Earnings25 testset-segmented mp3s.

Unlike iwslt+claude-to-pearmut/segment_audio.py this does not re-read the translation jsonl: the
annotations already carry `cut_start_in_clip` / `cut_end_in_clip`, i.e. the timing relative to the
~10-minute clip file of the audio behind the text shown to the annotator, so the cut is a direct
ffmpeg call. The source is mp3 (the stdlib
`wave` module cannot read it) and the output is mono mp3, which keeps the whole set at ~125 MB.

Writes OUT_DIR/<clip_id>/<clip_id>.<NNNN>.mp3, matching the `segment_filename` that
normalize_annotations.py puts in the records.

Usage:
    python cut_clips.py annotations.jsonl clips \
        --audio-dir ../../earnings25_raw/earnings-25/testset-segmented/audio
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def cut(job):
    src, dst, start, dur = job
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
         "-ac", "1", "-b:a", "64k", str(dst)],
        capture_output=True, text=True,
    )
    return dst, r.returncode, r.stderr.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="annotations.jsonl from normalize_annotations.py")
    ap.add_argument("out_dir")
    ap.add_argument("--audio-dir", required=True, help="dir with <clip_id>.mp3 (testset-segmented/audio)")
    ap.add_argument("--pad", type=float, default=0.3, help="seconds of context on each side (default: 0.3)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    args = ap.parse_args()

    audio_dir, out_dir = Path(args.audio_dir), Path(args.out_dir)
    jobs, missing, skipped = [], set(), 0
    for line in open(args.annotations, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        src = audio_dir / f"{rec['clip_id']}.mp3"
        if not src.exists():
            missing.add(src.name)
            continue
        dst = out_dir / Path(rec["segment_filename"])
        if dst.exists() and not args.force:
            skipped += 1
            continue
        # cut_*_in_clip is seg_*_in_clip widened where the stored ASR text runs past the segment
        lo = rec.get("cut_start_in_clip", rec["seg_start_in_clip"])
        hi = rec.get("cut_end_in_clip", rec["seg_end_in_clip"])
        start = max(0.0, lo - args.pad)
        dur = (hi + args.pad) - start
        jobs.append((src, dst, start, dur))

    for name in sorted(missing):
        print(f"WARNING: missing source audio {name}", file=sys.stderr)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, (dst, code, err) in enumerate(pool.map(cut, jobs), 1):
            if code != 0:
                failed += 1
                print(f"WARNING: ffmpeg failed on {dst}: {err}", file=sys.stderr)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clips", file=sys.stderr)

    print(f"{len(jobs)} cut, {skipped} already present, {failed} failed -> {out_dir}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
