#!/usr/bin/env python3
"""
Latency (LongYAAL etc.) of streaming outputs with OmniSTEval, with offline Canary output as the reference
segmentation.

    pip install OmniSTEval
    python eval_latency.py --ref-dir canary/ --hyp-dir nemo_streaming/fr/asr-low+EuroLLM-9B -l fr -o eval/fr-low
    python eval_latency.py --ref-dir canary/ --hyp-dir nemo_streaming/asr/low -l en -o eval/asr-low

Inputs, matched by file name <docid>.<lang>.jsonl in both directories:
  --ref-dir   offline Canary output (run_canary.py): one line per segment with "segment" (text) and
              "start"/"end" (seconds in the source audio); "words" are not needed by OmniSTEval.
              For a translation the Canary output in that target language, for ASR the transcript.
  --hyp-dir   streaming output of nemo_streaming_{parakeet,llm,cascade}.py: one line per step with the
              confirmed increment "text" and "emission_time" (seconds). "unconfirmed_text" is ignored:
              only confirmed text counts as emitted. The older format (with "words") works too.

What the script does:
  1. converts the inputs into OmniSTEval's longform inputs, written to <outdir>/omnisteval_input/:
       segmentation.json  [{"wav": docid, "offset": start, "duration": end - start}, ...]  (from Canary)
       references.txt     the Canary segment texts, one per line, in the same order
       hypothesis.jsonl   {"source": docid, "prediction": "...", "delays": [ms per word], "source_length": ms}
                          (prediction = all confirmed text in order; each word gets the emission_time of
                          the line that confirmed it)
  2. runs `omnisteval longform`, which aligns the hypothesis words to the reference segments
     (SoftSegmenter) and computes LongYAAL, LongAL, LongLAAL, LongAP, LongDAL. Report and scores.tsv go
     to <outdir>.

Notes:
  * Word times (--emission): "confirmed" (default) is the emission_time of the line that confirmed the word,
    i.e. the latency of showing only confirmed text. "first-stable" is the emission_time of the earliest
    line from which the word and all previous words stay unchanged in the hypothesis (confirmed text plus
    "unconfirmed_text") in every later line until the end of the output, i.e. the latency of a display that
    also shows the unconfirmed tail, counted from the moment it would never have to be corrected. Matching
    is exact (a change of punctuation or case counts as a change). The words and the resegmentation are the
    same in both modes, only their times differ.
  * Computation-unaware latency only: the streaming outputs have no computation-aware timestamps, so no
    "elapsed" field is written and OmniSTEval reports only the (CU) metrics.
  * Quality metrics are off by default (--quality turns them on): the reference is Canary's output, not a
    human reference, so BLEU/chrF against it measure agreement with Canary, not quality.
  * zh/ja are evaluated at character level (--char-level is set automatically), other languages at word
    level with Moses tokenization for the alignment.
  * Recording end: YAAL ignores words emitted at or after the end of the recording, and skips a segment
    whose first word comes that late. OmniSTEval takes the end of the last reference segment as the
    recording end. With Canary segments that is the end of the last speech, so output the system emits
    afterwards (e.g. the final flush after the end-of-utterance pause) is not counted, and a last segment
    whose words all come then drops out of the average. --recording-end audio counts output up to the end
    of the audio instead: it adds one empty reference segment per document from the end of the last
    Canary segment to the end of the audio (the empty segments show up as "empty predictions" in the
    report and do not enter the latency average). After the run, the script reports how much output was
    ignored for this reason.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CHAR_LEVEL_LANGS = ("zh", "ja")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{n}: not valid JSON ({e})")


def load_reference(path):
    """Canary offline JSONL -> [(start_s, end_s, text)] sorted by start."""
    segs = []
    for rec in read_jsonl(path):
        if "start" not in rec or "end" not in rec:
            sys.exit(f"{path}: a line has no 'start'/'end' (segment timestamps): {rec}")
        text = rec.get("segment")
        if text is None:  # fall back to the words
            text = " ".join(w.get("word", "") for w in rec.get("words", []))
        text = " ".join(str(text).split())  # one line, normalized spaces
        segs.append((float(rec["start"]), float(rec["end"]), text))
    segs.sort(key=lambda s: s[0])
    return segs


def to_units(text, char_level):
    return list("".join(text.split())) if char_level else text.split()


def load_hypothesis(path, char_level, emission="confirmed"):
    """Streaming JSONL -> (prediction, delays_ms, last_end_s).

    emission="confirmed": a word's time is the emission_time of the line whose "text" confirmed it.
    emission="first-stable": a word's time is the emission_time of the earliest line from which the word
    and all previous words stay unchanged in the hypothesis (confirmed text + "unconfirmed_text") in every
    later line until the end of the output. Never later than the confirmation time.
    """
    units, conf_delays, last_end = [], [], 0.0
    snapshots = []  # (time_ms, number of confirmed units so far, unconfirmed tail units) per line
    for ev in read_jsonl(path):
        if "end" in ev:
            last_end = max(last_end, float(ev["end"]))
        if "emission_time" not in ev:
            if (ev.get("text") or "").strip():
                sys.exit(f"{path}: a line with text has no 'emission_time': {ev}")
            continue
        t_ms = float(ev["emission_time"]) * 1000.0
        new = to_units(ev.get("text") or "", char_level)
        units += new
        conf_delays += [t_ms] * len(new)
        snapshots.append((t_ms, len(units), to_units(ev.get("unconfirmed_text") or "", char_level)))
    prediction = "".join(units) if char_level else " ".join(units)
    if emission == "confirmed":
        return prediction, conf_delays, last_end

    # Confirmed text is append-only, so it is always a prefix of the final output: the common prefix of a
    # line's hypothesis with the final output is the confirmed part plus the agreeing start of the tail.
    lcps = []
    for _, n_conf, tail in snapshots:
        k = 0
        while k < len(tail) and n_conf + k < len(units) and tail[k] == units[n_conf + k]:
            k += 1
        lcps.append(n_conf + k)
    # suffix minimum: from line j on, the first suff[j] units never change again (non-decreasing in j)
    suff, m = [0] * len(lcps), len(units)
    for j in range(len(lcps) - 1, -1, -1):
        m = min(m, lcps[j])
        suff[j] = m
    delays, i = [], 0
    for j, (t_ms, _, _) in enumerate(snapshots):
        while i < suff[j]:
            delays.append(min(t_ms, conf_delays[i]))
            i += 1
    delays += conf_delays[len(delays):]  # (not reached: the last line confirms everything)
    return prediction, delays, last_end


def audio_duration(audio_dir, docid):
    if not audio_dir:
        return None
    for p in sorted(Path(audio_dir).glob(docid + ".*")):
        try:
            import soundfile as sf
            return float(sf.info(str(p)).duration)
        except Exception:  # noqa: BLE001
            try:
                import librosa
                return float(librosa.get_duration(path=str(p)))
            except Exception:  # noqa: BLE001
                continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path, required=True,
                    help="directory with offline Canary outputs <docid>.<lang>.jsonl (reference segmentation)")
    ap.add_argument("--hyp-dir", type=Path, required=True,
                    help="directory with streaming outputs <docid>.<lang>.jsonl")
    ap.add_argument("-l", "--lang", required=True, help="language code of the evaluated output, e.g. en, de, cs, fr, zh")
    ap.add_argument("-o", "--outdir", type=Path, required=True, help="output directory for inputs, report and scores")
    ap.add_argument("--docs", nargs="+", help="only these docids (default: all hypothesis files in --hyp-dir)")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="only the first N documents (sorted by docid), for debugging: OmniSTEval's alignment is "
                    "slow on long documents")
    ap.add_argument("--skip-missing", action="store_true",
                    help="skip documents without a reference file instead of stopping")
    ap.add_argument("--audio-dir", type=Path,
                    help="audio files <docid>.* to read the recording length for source_length (default: the "
                    "last 'end' in the hypothesis, which is the audio length for outputs of these scripts)")
    ap.add_argument("--char-level", action="store_true",
                    help="character-level alignment and latency (automatic for zh and ja)")
    ap.add_argument("--quality", action="store_true",
                    help="also compute BLEU/chrF against the Canary output (agreement with Canary, not quality)")
    ap.add_argument("--emission", choices=["confirmed", "first-stable"], default="confirmed",
                    help="time of a hypothesis word: when it was confirmed (\"text\", default), or the earliest "
                    "line from which it and all previous words stay unchanged in the hypothesis (confirmed + "
                    "unconfirmed text) until the end of the output (see the notes above)")
    ap.add_argument("--recording-end", choices=["segments", "audio"], default="segments",
                    help="end of the recording for LongYAAL: end of the last reference segment (default, "
                    "OmniSTEval's behaviour) or the end of the audio (see the notes above)")
    ap.add_argument("--convert-only", action="store_true", help="write the OmniSTEval inputs but do not run it")
    args = ap.parse_args()

    lang = args.lang
    char_level = args.char_level or lang in CHAR_LEVEL_LANGS
    suffix = f".{lang}.jsonl"

    if args.docs:
        docids = list(args.docs)
    else:
        docids = sorted(p.name[: -len(suffix)] for p in args.hyp_dir.glob("*" + suffix))
    if not docids:
        sys.exit(f"no hypothesis files *{suffix} in {args.hyp_dir}")
    if args.limit:
        docids = docids[: args.limit]
        log(f"--limit {args.limit}: evaluating {len(docids)} document(s) only")

    seg_out, ref_lines, hyp_lines = [], [], []
    padding = set()  # indices of the empty segments added for --recording-end audio
    n_words = 0
    for docid in docids:
        hyp_path = args.hyp_dir / f"{docid}{suffix}"
        ref_path = args.ref_dir / f"{docid}{suffix}"
        if not hyp_path.exists():
            sys.exit(f"missing hypothesis {hyp_path}")
        if not ref_path.exists():
            if args.skip_missing:
                log(f"skip {docid}: no reference {ref_path}")
                continue
            sys.exit(f"missing reference {ref_path} (use --skip-missing to skip such documents)")

        segs = load_reference(ref_path)
        if not segs:
            log(f"skip {docid}: empty reference {ref_path}")
            continue
        prediction, delays, last_end = load_hypothesis(hyp_path, char_level, args.emission)

        for start, end, text in segs:
            # OmniSTEval needs a positive duration (source_length) for every segment
            seg_out.append({"wav": docid, "offset": round(start, 3), "duration": round(max(end - start, 0.01), 3)})
            ref_lines.append(text)

        duration = audio_duration(args.audio_dir, docid) or max(last_end, segs[-1][1])
        if args.recording_end == "audio" and duration - segs[-1][1] > 0.01:
            padding.add(len(seg_out))
            seg_out.append({"wav": docid, "offset": round(segs[-1][1], 3),
                            "duration": round(duration - segs[-1][1], 3)})
            ref_lines.append("")
        hyp_lines.append({"source": docid, "prediction": prediction, "delays": delays,
                          "source_length": round(duration * 1000.0, 1)})
        n_words += len(delays)
        log(f"{docid}: {len(segs)} reference segments, {len(delays)} hypothesis "
            f"{'characters' if char_level else 'words'}, {duration:.1f}s")

    if not hyp_lines:
        sys.exit("nothing to evaluate")

    inp = args.outdir / "omnisteval_input"
    inp.mkdir(parents=True, exist_ok=True)
    seg_file, ref_file, hyp_file = inp / "segmentation.json", inp / "references.txt", inp / "hypothesis.jsonl"
    seg_file.write_text(json.dumps(seg_out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    ref_file.write_text("".join(t + "\n" for t in ref_lines), encoding="utf-8")
    with open(hyp_file, "w", encoding="utf-8") as f:
        for h in hyp_lines:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    log(f"{len(hyp_lines)} documents, {len(ref_lines)} reference segments, {n_words} hypothesis units -> {inp}")

    cmd = [
        "longform",
        "--speech_segmentation", str(seg_file),
        "--ref_sentences_file", str(ref_file),
        "--hypothesis_file", str(hyp_file),
        "--hypothesis_format", "jsonl",
        "--char_level" if char_level else "--word_level",
        "--output_folder", str(args.outdir),
    ]
    if not char_level:
        cmd += ["--lang", lang]  # Moses tokenization for the alignment
    else:
        cmd += ["--bleu_tokenizer", "zh" if lang == "zh" else "char"]
    if not args.quality:
        cmd.append("--no_quality")

    runner = [sys.executable, "-c", "import sys; from omnisteval.cli import main; sys.exit(main())"]
    if args.convert_only:
        log("omnisteval " + " ".join(cmd))
        return 0
    try:
        import omnisteval  # noqa: F401
    except ImportError:
        sys.exit("OmniSTEval is not installed in this Python (pip install OmniSTEval). The inputs are in "
                 f"{inp}; run:\n  omnisteval " + " ".join(cmd))
    log("running: omnisteval " + " ".join(cmd))
    rc = subprocess.call(runner + cmd)
    if rc == 0:
        report_ignored(args.outdir / "instances.resegmented.jsonl", padding, args.recording_end)
    return rc


def report_ignored(path, padding, recording_end):
    """How much output YAAL ignored because it came at or after the recording end."""
    if not path.exists():
        return
    words = late = segs = skipped = 0
    for ins in read_jsonl(path):
        if ins["index"] in padding:
            continue
        d = ins.get("emission_cu") or []
        end = ins.get("time_to_recording_end")
        if not d or end is None:
            continue
        segs += 1
        words += len(d)
        late += sum(1 for x in d if x >= end)
        skipped += d[0] >= end
    if words:
        log(f"\nemitted at/after the recording end ({recording_end}), ignored by LongYAAL: {late} of {words} "
            f"words/characters ({100 * late / words:.1f}%); segments left out entirely: {skipped} of {segs}"
            + ("" if recording_end == "audio" or not late else "  (see --recording-end audio)"))


if __name__ == "__main__":
    sys.exit(main())