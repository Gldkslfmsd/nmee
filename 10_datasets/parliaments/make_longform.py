#!/usr/bin/env python3
"""
Build long-form "speeches" from the segment-level output of download_eurospeech.py /
download_voxpopuli.py.

Within each session, segments are sorted by their start time in the original recording
and chained into one speech while
  * the speaker stays the same (VoxPopuli: speaker_id; EuroSpeech has no speaker
    information, so there only the gap rule applies), and
  * the gap between the end of one segment and the start of the next is < --max-gap
    seconds (default 1.0). The gap is filled with silence of its exact length;
    overlapping segments are trimmed so no audio is duplicated.
A larger gap or a speaker change starts a new speech.

Segment timing:
  * EuroSpeech: start_seconds / end_seconds from the manifest.
  * VoxPopuli: not in the Hub data; taken from Meta's original annotations
    (https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_<lang>.tsv.gz,
    fields session_id, id_, vad=[(start, end), ...]), joined on
    audio_id == f"{session_id}-{id_}". Downloaded automatically into
    ROOT/_annotations unless --annotations-dir points elsewhere. Note that a VoxPopuli
    segment's audio is the concatenation of its VAD chunks, so short internal pauses
    are already removed; the segment's span is (first vad start, last vad end).

Output:
    OUT/audio/<config>/<split>/<session>/<speech_id>.wav     16-bit PCM, mono
    OUT/manifests/<config>/<split>.jsonl                      one line per speech

Speeches can be filtered with --min-segments and --min-duration (seconds of long-form
audio, gaps included); dropped speeches get no audio file.

Statistics (OUT/stats.txt, human-readable; OUT/stats.json, per config/split and ALL):
inserted silence gaps (< --max-gap) in kept speeches with a gap-length histogram,
trimmed overlaps, why speeches were split, and the dropped speeches (count, hours,
share of the audio, reason, duration histogram). --stats-only computes just these.

Each speech record: speech_id, config, split, session, speaker_id, language,
audio_filepath (relative to OUT), duration, orig_start, orig_end, n_segments,
the concatenated text fields, and "segments": [{id, offset, duration, orig_start,
orig_end, gap_before, <text fields>}] where offset/duration locate the segment
inside the long-form audio.

Splits are processed independently (a speech never crosses splits).

Requirements: pip install numpy soundfile
"""

import argparse
import csv
import gzip
import json
import sys
import urllib.request
from ast import literal_eval
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from utils import SPLITS, safe

VOX_ANN_URL = "https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_{lang}.tsv.gz"
TEXT_FIELDS = {
    "eurospeech": ["human_transcript", "asr_transcript"],
    "voxpopuli": ["raw_text", "normalized_text"],
}


# --------------------------------------------------------------------------- segment normalisation

def load_vox_annotations(lang: str, ann_dir: Path) -> dict:
    """audio_id -> (session_id, start, end) from Meta's asr_<lang>.tsv.gz."""
    path = ann_dir / f"asr_{lang}.tsv.gz"
    if not path.exists():
        ann_dir.mkdir(parents=True, exist_ok=True)
        url = VOX_ANN_URL.format(lang=lang)
        print(f"downloading {url}", file=sys.stderr)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    csv.field_size_limit(sys.maxsize)
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="|"):
            vad = literal_eval(r["vad"])
            if not vad:
                continue
            out[f"{r['session_id']}-{r['id_']}"] = (r["session_id"], float(vad[0][0]), float(vad[-1][1]))
    return out


def normalise(dataset: str, recs: list[dict], ann: dict | None) -> tuple[list[dict], int]:
    """Map manifest records to a common segment dict. Returns (segments, n_unmatched)."""
    segs, missing = [], 0
    for r in recs:
        if dataset == "eurospeech":
            seg = dict(id=r["key"], session=r.get("video_id") or "unknown", speaker=None,
                       start=float(r["start_seconds"]), end=float(r["end_seconds"]))
        else:
            a = ann.get(r["audio_id"])
            if a is None:
                missing += 1
                continue
            seg = dict(id=r["audio_id"], session=a[0], speaker=r.get("speaker_id"), start=a[1], end=a[2])
        seg.update(audio=r["audio_filepath"], language=r.get("language"),
                   texts={k: r.get(k) or "" for k in TEXT_FIELDS[dataset]})
        segs.append(seg)
    return segs, missing


def chain(segs: list[dict], max_gap: float) -> tuple[list[list[dict]], list[tuple[str, float]]]:
    """Split one session's segments into speeches.
    Returns (speeches, breaks) where breaks[i] = (reason, gap) between speech i and i+1,
    reason being "speaker" (speaker change) or "gap" (gap >= max_gap)."""
    segs = sorted(segs, key=lambda s: (s["start"], s["end"]))
    speeches, breaks, cur = [], [], []
    for s in segs:
        if cur:
            gap = s["start"] - cur[-1]["end"]
            if s["speaker"] != cur[-1]["speaker"]:
                breaks.append(("speaker", gap))
            elif gap >= max_gap:
                breaks.append(("gap", gap))
            else:
                cur.append(s)
                continue
            speeches.append(cur)
            cur = []
        cur.append(s)
    if cur:
        speeches.append(cur)
    return speeches, breaks


# --------------------------------------------------------------------------- audio

def measure_speech(speech: list[dict], root: str, sr: int) -> dict:
    """Exact long-form duration and gap list of a speech, reading only audio headers."""
    import soundfile as sf

    pos, prev_end, gaps, overlaps, seg_frames = 0, None, [], [], 0
    for s in speech:
        info = sf.info(Path(root) / s["audio"])
        frames = info.frames if info.samplerate == sr else int(round(info.frames * sr / info.samplerate))
        seg_frames += frames
        gap = 0.0 if prev_end is None else s["start"] - prev_end
        if prev_end is not None:
            if gap > 0:
                pos += int(round(gap * sr))
                gaps.append(gap)
            elif gap < 0:
                frames -= min(frames, int(round(-gap * sr)))
                overlaps.append(-gap)
            else:
                gaps.append(0.0)
        pos += frames
        prev_end = s["end"] if prev_end is None else max(prev_end, s["end"])
    return {"duration": pos / sr, "segment_audio": seg_frames / sr, "gaps": gaps, "overlaps": overlaps}


def build_speech(speech: list[dict], speech_id: str, config: str, split: str, root: str, out: str,
                 write_audio: bool, sr: int) -> dict:
    import numpy as np
    import soundfile as sf

    root_p, out_p = Path(root), Path(out)
    session = speech[0]["session"]
    rel = Path("audio") / config / split / safe(session) / f"{safe(speech_id)}.wav"
    pieces, seg_meta = [], []
    pos = 0  # samples written so far
    prev_end = None
    for s in speech:
        x, file_sr = sf.read(root_p / s["audio"], dtype="int16", always_2d=True)
        if file_sr != sr:
            raise ValueError(f"{s['audio']}: sampling rate {file_sr} != {sr}")
        x = x.mean(axis=1).astype(np.int16) if x.shape[1] > 1 else x[:, 0]
        gap = 0.0 if prev_end is None else s["start"] - prev_end
        if gap > 0:
            n = int(round(gap * sr))
            pieces.append(np.zeros(n, dtype=np.int16))
            pos += n
        elif gap < 0:  # overlap with previous segment: drop the duplicated beginning
            x = x[min(len(x), int(round(-gap * sr))):]
        seg_meta.append({"id": s["id"], "offset": round(pos / sr, 3), "duration": round(len(x) / sr, 3),
                         "orig_start": s["start"], "orig_end": s["end"], "gap_before": round(gap, 3),
                         **s["texts"]})
        pieces.append(x)
        pos += len(x)
        prev_end = max(prev_end, s["end"]) if prev_end is not None else s["end"]

    if write_audio:
        dst = out_p / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, np.concatenate(pieces) if pieces else np.zeros(0, np.int16), sr, subtype="PCM_16")

    rec = {"speech_id": speech_id, "config": config, "split": split, "session": session,
           "speaker_id": speech[0]["speaker"], "language": speech[0]["language"],
           "audio_filepath": str(rel), "duration": round(pos / sr, 3),
           "orig_start": speech[0]["start"], "orig_end": prev_end, "n_segments": len(speech)}
    for k in speech[0]["texts"]:
        rec[k] = " ".join(t for t in (s["texts"][k].strip() for s in speech) if t)
    rec["segments"] = seg_meta
    return rec


# --------------------------------------------------------------------------- statistics

GAP_BINS = [0, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SHORT_BINS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 120, float("inf")]


def describe(values) -> dict:
    import numpy as np

    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"count": 0}
    p = np.percentile(v, [10, 25, 50, 75, 90, 99])
    return {"count": int(v.size), "total_s": round(float(v.sum()), 2), "mean": round(float(v.mean()), 3),
            "min": round(float(v.min()), 3), "p10": round(float(p[0]), 3), "p25": round(float(p[1]), 3),
            "median": round(float(p[2]), 3), "p75": round(float(p[3]), 3), "p90": round(float(p[4]), 3),
            "p99": round(float(p[5]), 3), "max": round(float(v.max()), 3)}


def histogram(values, bins) -> list[dict]:
    import numpy as np

    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return []
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((v >= lo) & (v < hi)).sum())
        label = "0 (adjacent)" if (lo, hi) == (0, 0.0001) else f"[{lo:g}, {hi:g})"
        out.append({"bin": label, "count": n, "pct": round(100 * n / v.size, 2)})
    return out


class Stats:
    """Raw values for one (config, split); merge() builds dataset-level totals."""

    def __init__(self):
        self.n_segments = 0
        self.n_unmatched = 0
        self.n_sessions = 0
        self.segment_audio_s = 0.0
        self.kept_dur, self.kept_nseg = [], []
        self.dropped = []          # (reason, duration, n_segments)
        self.gaps_kept, self.gaps_dropped = [], []
        self.overlaps_kept = []
        self.breaks = []           # (reason, gap)

    def merge(self, other: "Stats"):
        for k, v in vars(other).items():
            setattr(self, k, getattr(self, k) + v)

    def report(self, max_gap: float, min_segments: int, min_duration: float) -> dict:
        all_dur = self.kept_dur + [d for _, d, _ in self.dropped]
        drop_dur = [d for _, d, _ in self.dropped]
        total_s, kept_s = sum(all_dur), sum(self.kept_dur)
        gaps_pos = [g for g in self.gaps_kept if g > 0]
        silence_s = sum(gaps_pos)
        by_reason = {}
        for reason in ("min_segments", "min_duration"):
            ds = [d for r, d, _ in self.dropped if r == reason]
            by_reason[reason] = {"speeches": len(ds), "hours": round(sum(ds) / 3600, 3)}
        return {
            "settings": {"max_gap": max_gap, "min_segments": min_segments, "min_duration": min_duration},
            "segments": {"used": self.n_segments, "skipped_no_timing": self.n_unmatched, "sessions": self.n_sessions,
                         "hours_segment_audio": round(self.segment_audio_s / 3600, 3)},
            "speeches": {
                "before_filtering": len(all_dur), "kept": len(self.kept_dur), "dropped": len(self.dropped),
                "hours_before_filtering": round(total_s / 3600, 3), "hours_kept": round(kept_s / 3600, 3),
                "hours_dropped": round((total_s - kept_s) / 3600, 3),
                "hours_silence_inserted_before_filtering": round(
                    (sum(g for g in self.gaps_kept + self.gaps_dropped if g > 0)) / 3600, 3),
                "kept_duration_s": describe(self.kept_dur),
                "kept_segments_per_speech": describe(self.kept_nseg),
            },
            "speech_breaks": {
                "speaker_change": sum(1 for r, _ in self.breaks if r == "speaker"),
                f"gap_ge_{max_gap:g}s": sum(1 for r, _ in self.breaks if r == "gap"),
                "break_gap_s": describe([g for r, g in self.breaks if r == "gap"]),
            },
            "inserted_silence_in_kept_speeches": {
                "joins": len(self.gaps_kept) + len(self.overlaps_kept),
                "joins_with_silence": len(gaps_pos),
                "joins_adjacent_no_gap": len(self.gaps_kept) - len(gaps_pos),
                "joins_with_overlap_trimmed": len(self.overlaps_kept),
                "silence_total_s": round(silence_s, 2),
                "silence_pct_of_kept_audio": round(100 * silence_s / kept_s, 2) if kept_s else 0.0,
                "silence_per_speech_s": round(silence_s / len(self.kept_dur), 3) if self.kept_dur else 0.0,
                "gap_s": describe(gaps_pos),
                "gap_histogram_non_overlapping_joins": histogram(self.gaps_kept, GAP_BINS),
                "overlap_trimmed_s": describe(self.overlaps_kept),
                "joins_in_dropped_speeches": len(self.gaps_dropped),
                "silence_in_dropped_speeches_s": round(sum(g for g in self.gaps_dropped if g > 0), 2),
            },
            "dropped_speeches": {
                "count": len(self.dropped),
                "pct_of_speeches": round(100 * len(self.dropped) / len(all_dur), 2) if all_dur else 0.0,
                "hours": round(sum(drop_dur) / 3600, 3),
                "pct_of_audio": round(100 * sum(drop_dur) / total_s, 2) if total_s else 0.0,
                "by_reason": by_reason,
                "duration_s": describe(drop_dur),
                "segments_per_speech": describe([n for _, _, n in self.dropped]),
                "duration_histogram": histogram(drop_dur, SHORT_BINS),
            },
        }


def hm(hours: float) -> str:
    """1.234 -> '1h14m' (rounded to the nearest minute)."""
    m = int(round(hours * 60))
    return f"{m // 60}h{m % 60:02d}m"


def format_report(name: str, r: dict) -> str:
    sp, br, g, d = r["speeches"], r["speech_breaks"], r["inserted_silence_in_kept_speeches"], r["dropped_speeches"]
    gs, ds = g["gap_s"], d["duration_s"]
    L = [f"== {name} ==",
         f"segments {r['segments']['used']} ({hm(r['segments']['hours_segment_audio'])} of audio; "
         f"skipped, no timing: {r['segments']['skipped_no_timing']}), sessions {r['segments']['sessions']}",
         f"before filtering: {sp['before_filtering']} speeches, {hm(sp['hours_before_filtering'])} "
         f"(incl. {hm(sp['hours_silence_inserted_before_filtering'])} inserted silence)",
         f"after filtering:  {sp['kept']} speeches, {hm(sp['hours_kept'])} kept; "
         f"dropped {sp['dropped']} speeches, {hm(sp['hours_dropped'])}",
         f"breaks between speeches: speaker change {br['speaker_change']}, "
         + ", ".join(f"{k} {v}" for k, v in br.items() if k.startswith("gap_ge")),
         "",
         "-- inserted silence (kept speeches) --",
         f"joins {g['joins']}: with silence {g['joins_with_silence']}, adjacent {g['joins_adjacent_no_gap']}, "
         f"overlap trimmed {g['joins_with_overlap_trimmed']}",
         f"silence total {g['silence_total_s']:.1f} s = {g['silence_pct_of_kept_audio']:.2f}% of kept audio, "
         f"{g['silence_per_speech_s']:.2f} s per speech"]
    if gs["count"]:
        L.append(f"gap length (s): mean {gs['mean']}, median {gs['median']}, p10 {gs['p10']}, p90 {gs['p90']}, "
                 f"max {gs['max']}")
    for h in g["gap_histogram_non_overlapping_joins"]:
        L.append(f"  {h['bin']:>14s} {h['count']:8d}  {h['pct']:6.2f}%  {'#' * int(h['pct'] / 2)}")
    L += ["", "-- dropped speeches --",
          f"{d['count']} = {d['pct_of_speeches']:.2f}% of speeches, {hm(d['hours'])} = {d['pct_of_audio']:.2f}% of audio",
          "by reason: " + ", ".join(f"{k} {v['speeches']} ({hm(v['hours'])})" for k, v in d["by_reason"].items())]
    if ds["count"]:
        L.append(f"duration (s): mean {ds['mean']}, median {ds['median']}, p10 {ds['p10']}, p90 {ds['p90']}, "
                 f"max {ds['max']}; segments/speech median {d['segments_per_speech']['median']}")
    for h in d["duration_histogram"]:
        if h["count"]:
            L.append(f"  {h['bin']:>14s} {h['count']:8d}  {h['pct']:6.2f}%  {'#' * int(h['pct'] / 2)}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- main

def detect_dataset(root: Path) -> str:
    for p in (root / "manifests").glob("*/*.jsonl"):
        with open(p, encoding="utf-8") as f:
            line = f.readline()
        if line:
            r = json.loads(line)
            if "audio_id" in r:
                return "voxpopuli"
            if "video_id" in r:
                return "eurospeech"
    sys.exit(f"could not detect dataset type from {root}/manifests; pass --dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="output root of download_eurospeech.py / download_voxpopuli.py")
    ap.add_argument("--dataset", choices=["eurospeech", "voxpopuli"], default=None, help="default: auto-detect")
    ap.add_argument("--out", default=None, help="default: ROOT/longform")
    ap.add_argument("--configs", nargs="*", default=None, help="countries (EuroSpeech) or languages (VoxPopuli)")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--max-gap", type=float, default=1.0, help="gaps >= this (seconds) start a new speech")
    ap.add_argument("--min-segments", type=int, default=1, help="drop speeches with fewer segments")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="drop speeches whose long-form audio is shorter than this many seconds, e.g. 45")
    ap.add_argument("--no-audio", action="store_true", help="only write manifests and statistics")
    ap.add_argument("--stats-only", action="store_true", help="only compute statistics (no audio, no manifests)")
    ap.add_argument("--annotations-dir", default=None, help="VoxPopuli: dir with asr_<lang>.tsv.gz (default: ROOT/_annotations)")
    ap.add_argument("--sampling-rate", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    dataset = args.dataset or detect_dataset(root)
    out = Path(args.out) if args.out else root / "longform"
    ann_dir = Path(args.annotations_dir) if args.annotations_dir else root / "_annotations"
    sr = args.sampling_rate

    configs = sorted(p.name for p in (root / "manifests").iterdir() if p.is_dir())
    if args.configs:
        configs = [c for c in configs if c in set(args.configs)]
    if not configs:
        sys.exit("no matching configs found under ROOT/manifests")

    settings = (args.max_gap, args.min_segments, args.min_duration)
    total = Stats()
    reports, text = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for config in configs:
            ann = None
            for split in args.splits:
                src = root / "manifests" / config / f"{split}.jsonl"
                if not src.exists():
                    continue
                with open(src, encoding="utf-8") as f:
                    recs = [json.loads(l) for l in f if l.strip()]
                if dataset == "voxpopuli" and ann is None:
                    ann = load_vox_annotations(config, ann_dir)
                segs, missing = normalise(dataset, recs, ann)
                if missing:
                    print(f"[warn] {config}/{split}: {missing}/{len(recs)} segments have no timing in the "
                          f"annotations and are skipped", file=sys.stderr)

                st = Stats()
                st.n_segments, st.n_unmatched = len(segs), missing
                by_session = defaultdict(list)
                for s in segs:
                    by_session[s["session"]].append(s)
                st.n_sessions = len(by_session)

                candidates = []
                for session, ss in sorted(by_session.items()):
                    speeches, breaks = chain(ss, args.max_gap)
                    st.breaks += breaks
                    candidates += [(sp, f"{session}_{i:04d}") for i, sp in enumerate(speeches)]

                # exact durations from audio headers, so filtering and statistics are precise
                measures = list(ex.map(measure_speech, [sp for sp, _ in candidates],
                                       [str(root)] * len(candidates), [sr] * len(candidates), chunksize=64))
                jobs = []
                for (sp, sid), m in zip(candidates, measures):
                    reason = ("min_segments" if len(sp) < args.min_segments
                              else "min_duration" if m["duration"] < args.min_duration else None)
                    st.segment_audio_s += m["segment_audio"]
                    if reason:
                        st.dropped.append((reason, m["duration"], len(sp)))
                        st.gaps_dropped += m["gaps"]
                    else:
                        st.kept_dur.append(m["duration"])
                        st.kept_nseg.append(len(sp))
                        st.gaps_kept += m["gaps"]
                        st.overlaps_kept += m["overlaps"]
                        jobs.append((sp, sid))

                if not args.stats_only:
                    futs = [ex.submit(build_speech, sp, sid, config, split, str(root), str(out),
                                      not args.no_audio, sr) for sp, sid in jobs]
                    results = [f.result() for f in as_completed(futs)]
                    results.sort(key=lambda r: (r["session"], r["orig_start"]))
                    dst = out / "manifests" / config / f"{split}.jsonl"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                name = f"{config}/{split}"
                reports[name] = st.report(*settings)
                text.append(format_report(name, reports[name]))
                total.merge(st)
                sp = reports[name]["speeches"]
                print(f"{name}: {len(segs)} segments ({hm(st.segment_audio_s / 3600)}) -> "
                      f"{sp['before_filtering']} speeches ({hm(sp['hours_before_filtering'])}) -> "
                      f"kept {sp['kept']} ({hm(sp['hours_kept'])}), dropped {sp['dropped']} "
                      f"({hm(sp['hours_dropped'])})", file=sys.stderr)

    if not reports:
        sys.exit("no manifests found for the selected configs/splits")
    reports["ALL"] = total.report(*settings)
    text.insert(0, format_report("ALL", reports["ALL"]))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    with open(out / "stats.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print("\n" + text[0], file=sys.stderr)
    print(f"statistics: {out / 'stats.txt'} and {out / 'stats.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
