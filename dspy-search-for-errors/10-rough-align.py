#!/usr/bin/env python3
"""Rough time-based alignment of golden-truth English transcript segments and
all automatic recognition/translation segments for one recording.

For a given recording id (e.g. 432864953_105.96_704.14) the script reads:

  * the golden-truth reference line for the recording: <data-jsonl>
    (its English transcript defines the main segmentation)
  * every automatic language available: <out-dir>/<id>.<lang>.jsonl
    (en = automatic English recognition, cs, de, pl, sk, ... = automatic
    translations, whatever exists for this recording)

and writes <dest-dir>/<id>.roughaligned.json containing, for each golden
English segment, all automatic segments (including the automatic English
recognition) that overlap with it in time, together with an estimated
word-count overlap derived from the timestamps, e.g. "approx. the first 5
words (of the golden English transcript) correspond to approx. the last
3 words of the automatic segment".

Golden transcript words carry no timestamps, so they are spread
proportionally over the audio duration when estimating word ranges.

Only the Python standard library is used.

Usage:
    python3 rough-align.py 432864953_105.96_704.14 \
        [--out-dir DIR] [--dest-dir DIR] [--data-jsonl FILE] [--indent N]

The script processes exactly one recording, so many instances can run in
parallel (see rough-align.runner.sh).
"""

import argparse
import difflib
import json
import os
import re
import sys

EPS = 1e-6


def load_jsonl(path):
    segments = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    return segments


def find_golden_record(data_jsonl, rec_id):
    with open(data_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("id") == rec_id:
                return rec
    return None


def normalize_word(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def build_en_word_timeline(automatic_segments):
    """Concatenate every word of the automatic English ASR ('en') segments,
    in chronological order, keeping only words with plausible (non-point,
    non-decreasing) timestamps. Returns a list of (word, start, end)."""
    en_segments = automatic_segments.get("en") or []
    timeline = []
    for seg in sorted(en_segments, key=lambda s: s.get("start") if isinstance(s.get("start"), (int, float)) else 0.0):
        times = word_times(seg)
        if times is None:
            continue
        words = seg.get("words") or []
        for w, (start, end) in zip(words, times):
            text = w.get("word")
            if text:
                timeline.append((text, start, end))
    return timeline


def anchor_golden_word_times(golden_words, en_timeline, duration):
    """Estimate a real (start, end) timestamp for every golden-transcript
    word, anchored to the real per-word timestamps of the automatic English
    ASR ('en') output (which is transcribing the very same audio), instead
    of assuming golden words are evenly paced over the whole recording.

    Golden and ASR text mostly agree word-for-word (same speech), so a
    word-level sequence alignment (difflib, like a diff) between the two
    gives frequent real-time anchor points throughout the recording. Golden
    words that don't line up with any ASR word (ASR errors/omissions) are
    filled in by linear interpolation *between their nearest anchors*
    only -- not by spreading them over the whole file -- which keeps the
    error local instead of letting it drift for the rest of the recording.

    Returns a list of dicts {"start": s, "end": e, "anchored": bool}
    parallel to golden_words. Falls back to old-style whole-file
    proportional spread only if no anchors could be found at all (e.g. no
    usable 'en' ASR output for this recording).
    """
    n = len(golden_words)
    resolved = [None] * n  # each: (start, end, anchored)

    if en_timeline:
        g_norm = [normalize_word(w) for w in golden_words]
        e_norm = [normalize_word(w) for w, _, _ in en_timeline]
        sm = difflib.SequenceMatcher(None, g_norm, e_norm, autojunk=False)
        for tag, g0, g1, e0, e1 in sm.get_opcodes():
            if tag not in ("equal", "replace"):
                continue
            pair_n = min(g1 - g0, e1 - e0)
            for k in range(pair_n):
                _, e_start, e_end = en_timeline[e0 + k]
                resolved[g0 + k] = (e_start, e_end, True)

    # Fill any unresolved words by linear interpolation between the nearest
    # resolved neighbors on either side (falling back to the recording's
    # [0, duration] bounds past the first/last anchor).
    anchor_idx = [i for i, r in enumerate(resolved) if r is not None]
    if not anchor_idx:
        # No anchors anywhere: behave like the old whole-file proportional
        # spread so recordings without usable 'en' ASR still get *some*
        # (coarse) estimate rather than crashing.
        for i in range(n):
            pos = 0.0 if n <= 1 else (i + 0.5) / n * duration
            resolved[i] = (pos, pos, False)
    else:
        prev_i, prev_end = None, 0.0
        for i in range(n):
            if resolved[i] is not None:
                prev_i, prev_end = i, resolved[i][1]
                continue
            # find next anchor at or after i
            nxt_i = next((j for j in anchor_idx if j > (prev_i if prev_i is not None else -1) and j >= i), None)
            next_start = resolved[nxt_i][0] if nxt_i is not None else duration
            span_lo = prev_end
            span_hi = next_start if next_start >= prev_end else prev_end
            lo_idx = (prev_i + 1) if prev_i is not None else 0
            hi_idx = nxt_i if nxt_i is not None else n
            span_len = max(1, hi_idx - lo_idx)
            frac = (i - lo_idx + 0.5) / span_len
            pos = span_lo + frac * (span_hi - span_lo)
            resolved[i] = (pos, pos, False)

    return [{"start": s, "end": e, "anchored": a} for s, e, a in resolved]


def build_golden_segment(golden, fallback_end, automatic_segments):
    """Return a pseudo-segment dict for the golden English transcript.

    Golden words carry no timestamps of their own, so they are anchored to
    the real per-word timestamps of the automatic English ASR output (see
    anchor_golden_word_times), which is far more accurate than assuming a
    constant speech rate over the whole (often ~10 minute) recording. The
    segment spans the whole recording portion, i.e. [0, duration].
    """
    duration = None
    audio_info = golden.get("audio_info") or {}
    if isinstance(audio_info.get("duration_seconds"), (int, float)):
        duration = float(audio_info["duration_seconds"])
    if duration is None:
        duration = fallback_end
    transcript = golden.get("transcript") or ""
    golden_words = transcript.split()
    en_timeline = build_en_word_timeline(automatic_segments)
    word_times_est = anchor_golden_word_times(golden_words, en_timeline, duration)
    words = [
        {"word": w, "start": t["start"], "end": t["end"], "anchored": t["anchored"]}
        for w, t in zip(golden_words, word_times_est)
    ]
    return {"start": 0.0, "end": duration, "segment": transcript, "words": words}


def word_times(segment):
    """Return per-word (start, end) tuples if they look usable, else None.

    Usable means: every word has numeric start/end, starts are
    non-decreasing, and not all words are point events (start == end).
    """
    words = segment.get("words") or []
    if not words:
        return None
    times = []
    prev_start = None
    all_points = True
    for w in words:
        start, end = w.get("start"), w.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None
        if prev_start is not None and start < prev_start - EPS:
            return None
        prev_start = start
        if end < start:
            start, end = end, start
        if end > start + EPS:
            all_points = False
        times.append((start, end))
    if all_points:
        return None
    return times


def word_range_in_interval(segment, interval_start, interval_end):
    """Estimate which words (1-based inclusive range) of `segment` fall into
    the time interval [interval_start, interval_end].

    Word-level timestamps are used when plausible; otherwise words are spread
    proportionally over the segment duration.  Returns (first, last) 1-based,
    or None if no word can be attributed to the interval.
    """
    words = segment.get("words") or []
    n = len(words)
    if n == 0:
        return None

    times = word_times(segment)
    if times is not None:
        # midpoints of word intervals
        positions = [(s + e) / 2.0 for s, e in times]
    else:
        seg_start = segment.get("start")
        seg_end = segment.get("end")
        if not isinstance(seg_start, (int, float)) or not isinstance(
            seg_end, (int, float)
        ):
            return None
        duration = seg_end - seg_start
        if duration <= EPS:
            # all words share the same time instant
            if interval_start - EPS <= seg_start <= interval_end + EPS:
                return (1, n)
            return None
        positions = [seg_start + (i + 0.5) / n * duration for i in range(n)]

    inside = [i for i, p in enumerate(positions) if interval_start <= p <= interval_end]
    if inside:
        return (inside[0] + 1, inside[-1] + 1)

    # Nothing strictly inside: if the interval is real, attribute the single
    # nearest word so that a genuine overlap never yields an empty estimate.
    if interval_end - interval_start <= EPS:
        return None
    mid = (interval_start + interval_end) / 2.0
    nearest = min(range(n), key=lambda i: abs(positions[i] - mid))
    return (nearest + 1, nearest + 1)


def describe_golden_part(rng, n_words):
    a, b = rng
    if a == 1 and b == n_words:
        return "all %d words of the golden English transcript" % n_words
    if a == 1:
        return "the first %d words (of the golden English transcript)" % b
    if b == n_words:
        return "the last %d words (of the golden English transcript)" % (n_words - a + 1)
    return "the %dth till %dth word of the golden English transcript" % (a, b)


def describe_automatic_part(rng, n_words):
    c, d = rng
    if c == 1 and d == n_words:
        return "all words in the automatic segment"
    if c == 1:
        return "the first %d words of the automatic segment" % d
    if d == n_words:
        return "the last %d words of the automatic segment" % (n_words - c + 1)
    return "words %d till %d of the automatic segment" % (c, d)


def main():
    parser = argparse.ArgumentParser(
        description="Rough time-based alignment of one recording "
        "(golden English transcript vs. all automatic segments, "
        "including the automatic English recognition)."
    )
    parser.add_argument("recording_id", help="e.g. 432864953_105.96_704.14")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "input",
            "no-more-embarrassing-errors-dominik",
            "out",
        ),
        help="directory with <id>.<lang>.jsonl files",
    )
    parser.add_argument(
        "--data-jsonl",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "input",
            "no-more-embarrassing-errors-dominik",
            "earnings-25",
            "testset-segmented",
            "data.jsonl",
        ),
        help="golden-truth data.jsonl (segment info per recording)",
    )
    parser.add_argument(
        "--dest-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "roughaligned"
        ),
        help="where <id>.roughaligned.json files are written",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indent (0 = compact)")
    args = parser.parse_args()

    rec_id = args.recording_id
    if not os.path.exists(args.data_jsonl):
        sys.exit("ERROR: golden-truth data.jsonl not found: %s" % args.data_jsonl)

    # discover automatic languages (including the automatic English "en")
    automatic_langs = []
    for fname in sorted(os.listdir(args.out_dir)):
        if not fname.endswith(".jsonl"):
            continue
        stem, dot, lang = fname[:-len(".jsonl")].rpartition(".")
        if stem == rec_id:
            automatic_langs.append(lang)
    if not automatic_langs:
        sys.exit(
            "ERROR: no automatic <%s>.<lang>.jsonl files found in %s"
            % (rec_id, args.out_dir)
        )

    automatic_segments = {
        lang: load_jsonl(os.path.join(args.out_dir, "%s.%s.jsonl" % (rec_id, lang)))
        for lang in automatic_langs
    }

    # golden-truth reference (defines the main segmentation)
    golden = find_golden_record(args.data_jsonl, rec_id)
    if golden is None:
        sys.exit("ERROR: golden-truth record for %s not found in %s" % (rec_id, args.data_jsonl))

    # fallback end if the audio duration is unknown: the latest automatic end
    fallback_end = 0.0
    for segs in automatic_segments.values():
        for seg in segs:
            if isinstance(seg.get("end"), (int, float)):
                fallback_end = max(fallback_end, seg["end"])

    golden_seg = build_golden_segment(golden, fallback_end, automatic_segments)
    g_start, g_end = golden_seg["start"], golden_seg["end"]
    g_words = golden_seg["words"]
    g_n = len(g_words)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result = {
        "id": rec_id,
        "data_jsonl": os.path.relpath(args.data_jsonl, base_dir),
        "automatic_files": {
            lang: os.path.relpath(
                os.path.join(args.out_dir, "%s.%s.jsonl" % (rec_id, lang)), base_dir
            )
            for lang in automatic_langs
        },
        "languages": {
            "golden_source": "en",
            "automatic": automatic_langs,
        },
        "golden_reference": (
            {
                "transcript": golden.get("transcript"),
                "audio_file_path": golden.get("audio_file_path"),
                "audio_info": golden.get("audio_info"),
                # Real per-word timestamps for the golden transcript, anchored
                # to the automatic English ASR's own word-level timestamps
                # (see anchor_golden_word_times) rather than assumed to be
                # evenly spaced over the whole recording. "anchored": true
                # means this word was matched directly to an ASR word;
                # "anchored": false means its time was interpolated between
                # the nearest anchors.
                "words": g_words,
            }
        ),
        "segments": [],
    }

    per_lang_overlaps = {}
    for lang in automatic_langs:
        lang_overlaps = []
        for auto_idx, auto_seg in enumerate(automatic_segments[lang]):
            auto_start, auto_end = auto_seg.get("start"), auto_seg.get("end")
            if (
                not isinstance(g_start, (int, float))
                or not isinstance(g_end, (int, float))
                or not isinstance(auto_start, (int, float))
                or not isinstance(auto_end, (int, float))
            ):
                continue
            ov_start = max(g_start, auto_start)
            ov_end = min(g_end, auto_end)
            if ov_end - ov_start <= EPS:
                continue

            auto_words = auto_seg.get("words") or []
            auto_n = len(auto_words)

            g_range = word_range_in_interval(golden_seg, ov_start, ov_end)
            auto_range = word_range_in_interval(auto_seg, ov_start, ov_end)
            if g_range is None or auto_range is None:
                # fall back to proportional counts by duration
                g_cnt = max(
                    1, int(round((ov_end - ov_start) / max(g_end - g_start, EPS) * g_n))
                ) if g_n else 0
                auto_cnt = max(
                    1, int(round((ov_end - ov_start) / max(auto_end - auto_start, EPS) * auto_n))
                ) if auto_n else 0
                g_range = (1, min(g_cnt, g_n)) if g_n else None
                auto_range = (1, min(auto_cnt, auto_n)) if auto_n else None

            # Tight real-time bounds for exactly the words in each range
            # (not the whole segment's [start, end]), read straight from the
            # per-word timestamps -- golden's are anchored to the automatic
            # English ASR (see build_golden_segment), the automatic
            # segment's are its own ASR/MT word timestamps when usable.
            golden_word_time_range = None
            if g_range:
                golden_word_time_range = [
                    g_words[g_range[0] - 1]["start"],
                    g_words[g_range[1] - 1]["end"],
                ]
            automatic_word_time_range = None
            auto_times = word_times(auto_seg)
            if auto_range and auto_times:
                automatic_word_time_range = [
                    auto_times[auto_range[0] - 1][0],
                    auto_times[auto_range[1] - 1][1],
                ]

            entry = {
                "automatic_segment_index": auto_idx,
                "start": auto_start,
                "end": auto_end,
                "segment": auto_seg.get("segment"),
                "word_count": auto_n,
                "overlap_start": ov_start,
                "overlap_end": ov_end,
                "overlap_duration": round(ov_end - ov_start, 3),
                "golden_word_count_overlap": (
                    g_range[1] - g_range[0] + 1 if g_range else 0
                ),
                "automatic_word_count_overlap": (
                    auto_range[1] - auto_range[0] + 1 if auto_range else 0
                ),
                "golden_word_range": g_range,  # 1-based, inclusive
                "automatic_word_range": auto_range,  # 1-based, inclusive
                "golden_word_time_range": golden_word_time_range,  # [start, end] seconds
                "automatic_word_time_range": automatic_word_time_range,  # [start, end] seconds
            }
            if g_range and auto_range:
                entry["estimated_word_count_overlap"] = "approx. %s correspond to approx. %s" % (
                    describe_golden_part(g_range, g_n),
                    describe_automatic_part(auto_range, auto_n),
                )
            lang_overlaps.append(entry)
        if lang_overlaps:
            per_lang_overlaps[lang] = lang_overlaps

    result["segments"].append(
        {
            "golden_segment_index": 0,
            "start": g_start,
            "end": g_end,
            "transcript": golden.get("transcript"),
            "word_count": g_n,
            "automatic_overlapping_segments": per_lang_overlaps,
        }
    )

    os.makedirs(args.dest_dir, exist_ok=True)
    out_path = os.path.join(args.dest_dir, rec_id + ".roughaligned.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        indent = args.indent if args.indent > 0 else None
        json.dump(result, fh, ensure_ascii=False, indent=indent)
        if indent is not None:
            fh.write("\n")
    print("wrote %s (1 golden English segment with %d words, %d automatic languages)" % (
        out_path, g_n, len(automatic_langs)))


if __name__ == "__main__":
    main()
