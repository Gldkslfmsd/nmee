#!/usr/bin/env python3
"""Cut long-form audio into per-segment clips using timestamps from translation jsonl files.

For every <doc>.en.jsonl in --translations-dir (one JSON object per line with "segment" text and
"start"/"end" in seconds), reads <doc>.wav (or .mp3, .flac, ... see --audio-ext) from --audio-dir and writes
    OUT_DIR/<doc>/<doc>.<NNNN>.wav      (NNNN = 0-based line index in the jsonl)

Usage:
    python segment_audio.py OUT_DIR \
        --audio-dir ~/work/uedin/mtm26/nmee/iwslt26-cs-dev/audio \
        --translations-dir ~/work/uedin/mtm26/nmee/outputs/iwslt26-cs-dev \
        [--suffix .en.jsonl] [--pad 0.2] [--how-many 3] [--debug-prints DEBUG_DIR] [-j JOBS] [--tmp-dir DIR]

--how-many N          process only the first N documents (sorted by name), for debugging
--debug-prints DIR    write DIR/<doc>.txt with "start<TAB>end<TAB>text" per segment, plus DIR/stats.txt
                      with statistics (durations, shortest segments, empty texts, overlaps, segments
                      outside the audio); the statistics are also printed to stderr
-j, --jobs N          number of parallel workers, one document per worker (default: number of CPUs)
--tmp-dir DIR         where to put temporary decoded WAVs (default: system temp dir, $TMPDIR)

Clips are always written as WAV in the source's original sample format (sample rate, bit depth,
int/float, channels). Uncompressed WAV input (PCM, float, A-law/mu-law, WAVE_FORMAT_EXTENSIBLE) is cut by
copying raw bytes, with the source's fmt chunk copied verbatim into every clip. Other formats (mp3, flac,
...) are first decoded ONCE per document to a temporary WAV in the decoder's native sample format with
`ffmpeg` (or `sox` if ffmpeg is not on PATH), which is then cut the same way. This avoids spawning one
process per segment, which re-probes and often re-decodes the file from the start every time.
The work is subprocess- and I/O-bound, so a thread pool is used.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# PCM, IEEE float, A-law, mu-law, extensible: all have a fixed number of bytes per frame
CUTTABLE_TAGS = {1, 3, 6, 7, 0xFFFE}


def wav_layout(path):
    """Walk the RIFF chunks of a WAV file.
    Returns (fmt_chunk, sample_rate, block_align, data_offset, n_frames)."""
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(12)
        if hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        fmt = None
        while True:
            ch = f.read(8)
            if len(ch) < 8:
                raise ValueError("no data chunk")
            cid, size = ch[:4], struct.unpack("<I", ch[4:])[0]
            if cid == b"fmt ":
                fmt = f.read(size)
                f.seek(size % 2, 1)
            elif cid == b"data":
                if fmt is None or len(fmt) < 16:
                    raise ValueError("missing or short fmt chunk")
                off = f.tell()
                size = min(size, fsize - off)   # streamed files may have a bogus size
                sr = struct.unpack("<I", fmt[4:8])[0]
                ba = struct.unpack("<H", fmt[12:14])[0]
                if sr <= 0 or ba <= 0:
                    raise ValueError("invalid sample rate or block align")
                return fmt, sr, ba, off, size // ba
            else:
                f.seek(size + size % 2, 1)


def wave_readable(path):
    """True if the WAV is uncompressed and can be cut by copying bytes, keeping its exact format."""
    if path.suffix.lower() != ".wav":
        return False
    try:
        fmt = wav_layout(path)[0]
        tag = struct.unpack("<H", fmt[:2])[0]
        if tag in CUTTABLE_TAGS:
            return True
        reason = f"format tag {tag:#x} is compressed"
    except (OSError, ValueError, struct.error) as e:
        reason = str(e)
    print(f"{path}: can't cut directly ({reason}), decoding once with an external tool", file=sys.stderr)
    return False


def audio_duration(path):
    if path.suffix.lower() == ".wav":
        try:
            _, sr, _, _, n = wav_layout(path)
            return n / sr
        except (OSError, ValueError, struct.error):
            pass
    for cmd in (["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                ["soxi", "-D", str(path)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
            return float(out.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
    return None


def write_wav(path, fmt, data, n_frames):
    """Write a WAV with the source's fmt chunk copied verbatim, so sample rate, bit depth,
    int/float and channel layout are exactly the original's."""
    tag = struct.unpack("<H", fmt[:2])[0]
    chunks = [b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"\0" * (len(fmt) % 2)]
    if tag != 1:   # non-PCM formats should carry a fact chunk with the frame count
        chunks.append(b"fact" + struct.pack("<II", 4, n_frames))
    chunks.append(b"data" + struct.pack("<I", len(data)))
    head = b"".join(chunks)
    with open(path, "wb") as o:
        o.write(b"RIFF" + struct.pack("<I", 4 + len(head) + len(data) + len(data) % 2) + b"WAVE")
        o.write(head)
        o.write(data)
        if len(data) % 2:
            o.write(b"\0")


def cut_with_wave(wav_path, segs, out_dir, stem, pad):
    """Cut a whole document in-process by copying raw frames (sequential reads of one file are fastest)."""
    fmt, sr, ba, off, n = wav_layout(wav_path)
    with open(wav_path, "rb") as f:
        for i, s in enumerate(segs):
            a = min(n, max(0, int((s["start"] - pad) * sr)))
            b = min(n, int((s["end"] + pad) * sr))
            f.seek(off + a * ba)
            data = f.read(max(0, b - a) * ba)
            write_wav(out_dir / f"{stem}.{i:04d}.wav", fmt, data, len(data) // ba)


PCM_CODEC = {"u8": "pcm_u8", "s16": "pcm_s16le", "flt": "pcm_f32le", "dbl": "pcm_f64le", "s64": "pcm_s64le"}


def pcm_codec_for(path):
    """PCM codec that holds the decoder's native sample format losslessly."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                              "stream=sample_fmt,bits_per_raw_sample", "-of", "json", str(path)],
                             capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL)
        st = json.loads(out.stdout)["streams"][0]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError):
        return "pcm_f32le"
    sfmt = st.get("sample_fmt", "").rstrip("p")   # planar variants (fltp, s16p) -> packed
    if sfmt == "s32":
        return "pcm_s24le" if str(st.get("bits_per_raw_sample")) == "24" else "pcm_s32le"
    return PCM_CODEC.get(sfmt, "pcm_f32le")


def decode_to_wav(tool, src, dst):
    """Decode a whole file once to WAV, keeping sample rate, channels and sample format."""
    if tool == "ffmpeg":
        # -nostdin: parallel ffmpeg processes would otherwise fight over the terminal's stdin
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
               "-vn", "-c:a", pcm_codec_for(src), "-f", "wav", str(dst)]
    else:
        cmd = ["sox", str(src), str(dst)]   # sox keeps the input's precision by default
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def cut_doc(wav, segs, out_dir, stem, pad, use_wave, tool, tmp_dir):
    """One job per document: cut directly, or decode once to a temp WAV and cut that."""
    if use_wave:
        cut_with_wave(wav, segs, out_dir, stem, pad)
        return
    fd, tmp = tempfile.mkstemp(prefix=f"{stem}.", suffix=".wav", dir=tmp_dir)
    os.close(fd)
    try:
        decode_to_wav(tool, wav, tmp)
        cut_with_wave(Path(tmp), segs, out_dir, stem, pad)
    finally:
        os.unlink(tmp)


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


# ---------------------------------------------------------------- per-document preparation

def prepare_doc(jf, suffix, audio_dir, exts, debug_dir):
    """Read segments, locate audio, decide the cutting method and gather debug info (runs in parallel)."""
    stem = jf.name[: -len(suffix)]
    wav = find_audio(audio_dir, stem, exts)
    segs = read_segments(jf)
    d = {"stem": stem, "wav": wav, "segs": segs, "use_wave": bool(wav) and wave_readable(wav)}
    if debug_dir:
        write_tsv(debug_dir / f"{stem}.txt", segs)
        audio_len = audio_duration(wav) if wav else None
        durs = [s["end"] - s["start"] for s in segs]
        probs = doc_problems(stem, segs, audio_len)
        alen = f"{audio_len:.1f}s" if audio_len is not None else "missing"
        covered = f" ({sum(x for x in durs if x > 0) / audio_len:.0%} covered)" if audio_len else ""
        d["durs"] = durs
        d["probs"] = probs
        d["doc_line"] = f"{stem}: audio {alen}{covered}; {fmt_stats(durs)}; {len(probs)} issue(s)"
    return d


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
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4,
                    help="number of parallel workers, one document each (default: number of CPUs)")
    ap.add_argument("--tmp-dir", default=None,
                    help="directory for temporary decoded WAVs (default: system temp dir / $TMPDIR)")
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

    ext_tool = "ffmpeg" if shutil.which("ffmpeg") else ("sox" if shutil.which("sox") else None)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        # phase 1: prepare all documents in parallel (map keeps the sorted order for the report)
        docs = list(ex.map(lambda jf: prepare_doc(jf, args.suffix, audio_dir, exts, debug_dir), files))

        # phase 2: one job per document
        jobs = []
        for d in docs:
            stem, wav, segs = d["stem"], d["wav"], d["segs"]
            if wav is None:
                print(f"WARNING: no audio for {stem} in {audio_dir} ({', '.join(exts)})", file=sys.stderr)
                continue
            if not d["use_wave"] and ext_tool is None:
                sys.exit(f"{wav}: needs ffmpeg or sox on PATH")
            out_dir = out_root / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            d["out_dir"] = out_dir
            if not segs:
                print(f"{stem}: 0 clips -> {out_dir}")
                continue
            jobs.append(d)
        # largest files first, so that the longest job does not start last
        jobs.sort(key=lambda d: d["wav"].stat().st_size, reverse=True)

        # phase 3: run everything, report each document when it finishes
        failed = 0
        futs = {ex.submit(cut_doc, d["wav"], d["segs"], d["out_dir"], d["stem"], args.pad,
                          d["use_wave"], ext_tool, args.tmp_dir): d for d in jobs}
        for f in as_completed(futs):
            d = futs[f]
            try:
                f.result()
                print(f"{d['stem']}: {len(d['segs'])} clips -> {d['out_dir']}")
            except Exception as e:
                failed += 1
                print(f"ERROR in {d['stem']}: {e}", file=sys.stderr)

    if debug_dir:
        all_durs, all_segs, all_probs, doc_lines = [], [], [], []
        for d in docs:
            all_durs += d["durs"]
            all_segs += [(s["end"] - s["start"], d["stem"], i, s) for i, s in enumerate(d["segs"])]
            all_probs += d["probs"]
            doc_lines.append(d["doc_line"])

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

    if failed:
        sys.exit(f"{failed} job(s) failed")


if __name__ == "__main__":
    main()