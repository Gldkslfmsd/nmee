#!/usr/bin/env python3
"""Speech translation with NVIDIA Canary over one or more audio files.

Writes one JSONL per (input file, source lang, target lang): one segment per
line, with the words belonging to that segment nested under "words".

Example:
    python canary_translate.py -t cs pl -o out/ audio/*.mp3
"""

import argparse
import contextlib
import json
import os
import socket
import soundfile as sf
import sys
import tempfile
from pathlib import Path

from nemo.collections.asr.models import ASRModel

SEG_KEYS = ("segment", "start_offset", "end_offset", "start", "end")
WORD_KEYS = ("word", "start_offset", "end_offset", "start", "end")


def silence_resource_tracker():
    """Suppress the ResourceTracker traceback printed at interpreter shutdown.

    `multiprocess` (dill's fork, pulled in by NeMo) touches an RLock internal
    that no longer exists in Python 3.12, and it does so from __del__ during
    finalisation. Python catches that itself and prints "Exception ignored in",
    so it cannot be caught at the call site -- the only way to hide it is to
    make the underlying _stop non-throwing.
    """
    for module in ("multiprocess.resource_tracker", "multiprocessing.resource_tracker"):
        try:
            tracker = __import__(module, fromlist=["ResourceTracker"]).ResourceTracker
        except (ImportError, AttributeError):
            continue
        original = tracker._stop

        def quiet_stop(self, *args, __original=original, **kwargs):
            try:
                return __original(self, *args, **kwargs)
            except Exception:
                return None

        tracker._stop = quiet_stop


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path,
                   help="input audio file(s)")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("."),
                   help="directory for the .jsonl outputs (default: cwd)")
    p.add_argument("-s", "--source-lang", default="en",
                   help="source language code (default: en)")
    p.add_argument("-t", "--target-lang", nargs="+", required=True,
                   help="target language code(s); one output file per code")
    p.add_argument("-m", "--model", default="nvidia/canary-1b-v2",
                   help="pretrained model name or path to a .nemo file")
    p.add_argument("-b", "--batch-size", type=int, default=1,
                   help="transcribe batch size; keep at 1 so that long-form "
                        "chunking stays enabled (default: 1)")
    p.add_argument("--overwrite", action="store_true",
                   help="regenerate outputs that already exist")
    p.add_argument("--no-lock", action="store_true",
                   help="do not take a per-output lock directory; use when "
                        "only one job is running over these outputs")
    return p.parse_args()


def subset(d, keys):
    """Project a timestamp dict onto the wanted keys, preserving order."""
    return {k: d[k] for k in keys if k in d}


def words_for_segment(seg, words):
    """Words contained in the segment, matched on frame offsets when present."""
    if "start_offset" in seg and "end_offset" in seg:
        lo, hi = seg["start_offset"], seg["end_offset"]
        key = ("start_offset", "end_offset")
    else:
        lo, hi = seg.get("start"), seg.get("end")
        key = ("start", "end")
    if lo is None or hi is None:
        return []
    out = []
    for w in words:
        ws, we = w.get(key[0]), w.get(key[1])
        if ws is None or we is None:
            continue
        if ws >= lo and we <= hi:
            out.append(subset(w, WORD_KEYS))
    return out


def ensure_mono(path, tmpdir, cache):
    """Canary takes single-channel audio; downmix anything wider to mono.

    Returns the path to feed the model: the original when it is already mono,
    otherwise a 16-bit WAV in `tmpdir`. Sample rate is left alone -- NeMo
    resamples to 16 kHz itself.
    """
    if path in cache:
        return cache[path]

    try:
        info = sf.info(str(path))
    except Exception as exc:
        print(f"warning: cannot probe {path} ({exc}); passing through",
              file=sys.stderr)
        cache[path] = path
        return path

    if info.channels == 1:
        cache[path] = path
        return path

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    out = Path(tmpdir) / f"{path.stem}.mono.wav"
    sf.write(str(out), mono, sr, subtype="PCM_16")
    print(f"{path.name}: {info.channels} channels, downmixed to mono",
          file=sys.stderr)
    cache[path] = out
    return out


@contextlib.contextmanager
def file_lock(path, enabled=True):
    """mkdir-based lock next to `path`, so several jobs can share a work list.

    Yields True if this process took the lock, False if someone else holds it.
    mkdir is atomic on POSIX filesystems, which is what makes this safe without
    any coordination between the workers.
    """
    if not enabled:
        yield True
        return

    lock = path.with_name(path.name + ".lock")
    try:
        os.mkdir(lock)
    except FileExistsError:
        yield False
        return

    owner = lock / "owner"
    try:
        owner.write_text(f"{socket.gethostname()} pid {os.getpid()}\n")
        yield True
    finally:
        try:
            owner.unlink()
        except OSError:
            pass
        try:
            os.rmdir(lock)
        except OSError:
            pass


def write_jsonl(hyp, path):
    """Write the hypothesis to `path`, via a temp file so that an interrupted
    run never leaves a partial .jsonl that the skip-if-exists check would
    later mistake for a finished one."""
    ts = getattr(hyp, "timestamp", None) or {}
    segments = ts.get("segment") or []
    words = ts.get("word") or []

    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        for seg in segments:
            rec = subset(seg, SEG_KEYS)
            rec["words"] = words_for_segment(seg, words)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

    return len(segments), len(words), (segments[-1].get("end") if segments else None)


def main():
    silence_resource_tracker()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        sys.exit("no such file(s): " + ", ".join(str(p) for p in missing))

    model = ASRModel.from_pretrained(model_name=args.model)

    warned_no_words = False
    mono_cache = {}
    tmpdir_ctx = tempfile.TemporaryDirectory(prefix="canary-mono-")

    with tmpdir_ctx as tmpdir:
        for inp in args.inputs:
            for tgt in args.target_lang:
                out = args.output_dir / f"{inp.stem}.{tgt}.jsonl"
                if out.exists() and not args.overwrite:
                    print(f"skip, exists: {out}", file=sys.stderr)
                    continue

                with file_lock(out, enabled=not args.no_lock) as acquired:
                    if not acquired:
                        print(f"skip, locked by another job: {out}",
                              file=sys.stderr)
                        continue
                    # another worker may have finished it between the check
                    # above and the lock being taken
                    if out.exists() and not args.overwrite:
                        print(f"skip, exists: {out}", file=sys.stderr)
                        continue

                    audio = ensure_mono(inp, tmpdir, mono_cache)

                    hyps = model.transcribe(
                        [str(audio)],
                        source_lang=args.source_lang,
                        target_lang=tgt,
                        timestamps=True,
                        batch_size=args.batch_size,
                    )
                    # some NeMo versions return (best, all) hypotheses
                    if isinstance(hyps, tuple):
                        hyps = hyps[0]

                    n_seg, n_word, last_end = write_jsonl(hyps[0], out)

                print(f"{out}: {n_seg} segments, {n_word} words, "
                      f"last end {last_end}", file=sys.stderr)
                sys.stderr.flush()
                if n_seg and not n_word and not warned_no_words:
                    print("note: no word-level timestamps returned; "
                          "\"words\" will be empty for every segment",
                          file=sys.stderr)
                    warned_no_words = True


if __name__ == "__main__":
    main()
