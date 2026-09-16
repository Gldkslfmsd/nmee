#!/usr/bin/env python3
"""Merge per-model divergence findings (./25-divergencies-by-llms/*.json,
produced by claude-find-bad-translation-divergencies.py) into a single list
of error items, one per source file, suitable for manual review.

No LLM calls are made here -- models are merged with a purely mechanical
heuristic: locate each reported error's `source_words` within the golden
transcript (via the same difflib word-alignment trick the divergence
finder itself uses for reconstruction), then merge/cluster error
occurrences -- across all available models -- whose golden-word ranges
overlap or nearly touch. Each resulting cluster becomes one output item.

For each item we report:
  - source_file, golden_segment_index, golden_word_range
  - start/end (seconds), reconstructed from the ORIGINAL roughaligned data
    (not trusted from any model)
  - golden_transcript: the exact golden-truth text for that word range,
    read fresh from roughaligned/*.json (authoritative, not any model's
    possibly-imperfect quote)
  - models_reporting / num_models_reporting
  - explanations: every contributing model's error_class/source_words/
    explanation, tagged with which model said it
  - automatic_outputs: per language (en/de/cs/pl/sk), the full original
    automatic text overlapping this item's timespan (read fresh from
    roughaligned data), plus "good" and "bad" renderings as judged by each
    contributing model

Output: ./merged_errors.json (full detail) and ./merged_errors.csv (flat
summary for quick spreadsheet triage).

Usage
-----
    ./venv/bin/python 30-merge-annotations.py
    ./venv/bin/python 30-merge-annotations.py --roughaligned-dir 15-roughaligned --divergencies-dir 25-divergencies-by-llms
"""

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent

TARGET_AUTOMATIC_LANGS = ["en", "de", "cs", "pl", "sk"]

# How many golden words apart two error occurrences' word ranges may be and
# still be merged into the same cluster (absorbs small alignment jitter
# between models/occurrences that are really about the same spot).
GAP_TOLERANCE_WORDS = 3


# ---------------------------------------------------------------------------
# Word-range helpers (same approach as claude-find-bad-translation-divergencies.py)
# ---------------------------------------------------------------------------


def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def align_words(golden_words: list[str], pos: int, query_words: list[str], slack: int) -> tuple[int, int]:
    """Best-effort: locate query_words within golden_words at or after
    `pos`, searching a window of len(query_words)+slack words. Returns a
    0-based half-open (start, end) range."""
    if not query_words:
        return pos, pos + 1
    window_end = min(len(golden_words), pos + len(query_words) + slack)
    window = golden_words[pos:window_end]
    sm = difflib.SequenceMatcher(None, window, query_words, autojunk=False)
    match = sm.find_longest_match(0, len(window), 0, len(query_words))
    if match.size == 0:
        start_idx = pos
    else:
        start_idx = pos + max(0, match.a - match.b)
    end_idx = min(max(start_idx + len(query_words), start_idx + 1), len(golden_words))
    return start_idx, end_idx


def locate_error_word_range(
    golden_words: list[str], chunk_range: tuple[int, int], source_words: str
) -> tuple[int, int]:
    """Find a specific error's source_words within its containing chunk's
    (1-based inclusive) golden_word_range. Falls back to the whole chunk
    range if the quote can't be confidently located."""
    chunk_start, chunk_end = chunk_range
    pos0 = max(0, chunk_start - 1)
    query_words = source_words.split()
    if not query_words:
        return chunk_range
    window_len = max(1, chunk_end - chunk_start + 1)
    slack = max(window_len, len(query_words))
    start_idx, end_idx = align_words(golden_words, pos0, query_words, slack)
    start_idx = max(start_idx, pos0)
    end_idx = min(end_idx, chunk_end)
    if end_idx <= start_idx:
        return chunk_range
    return (start_idx + 1, end_idx)


# ---------------------------------------------------------------------------
# Loading roughaligned source data
# ---------------------------------------------------------------------------


class GoldenSegmentData:
    __slots__ = ("golden_words", "lang_segments")

    def __init__(self, golden_words: list[str], lang_segments: dict[str, list[dict]]):
        self.golden_words = golden_words
        self.lang_segments = lang_segments


def load_roughaligned(path: Path) -> dict[int, GoldenSegmentData]:
    """Map golden_segment_index -> GoldenSegmentData for one roughaligned file."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    available_langs = [l for l in TARGET_AUTOMATIC_LANGS if l in doc.get("languages", {}).get("automatic", [])]
    out = {}
    for gseg in doc.get("segments", []):
        overlapping = gseg.get("automatic_overlapping_segments", {})
        lang_segments = {}
        for lang in available_langs:
            segs = sorted(overlapping.get(lang, []), key=lambda s: s["automatic_segment_index"])
            lang_segments[lang] = [
                {
                    "automatic_segment_index": s["automatic_segment_index"],
                    "start": s["start"],
                    "end": s["end"],
                    "segment": s["segment"],
                    "golden_word_range": tuple(s["golden_word_range"]) if s.get("golden_word_range") else None,
                }
                for s in segs
            ]
        out[gseg["golden_segment_index"]] = GoldenSegmentData(
            golden_words=gseg["transcript"].split(), lang_segments=lang_segments
        )
    return out


# ---------------------------------------------------------------------------
# Discover model output files
# ---------------------------------------------------------------------------


def discover_model_outputs(divergencies_dir: Path, roughaligned_stems: set[str]) -> dict[str, dict[str, Path]]:
    """Map roughaligned-stem -> {model_name: output_path}."""
    result: dict[str, dict[str, Path]] = {}
    for p in sorted(divergencies_dir.glob("*.json")):
        # filenames are "<roughaligned-stem>.<model>.json"
        name = p.name[: -len(".json")]
        matched_stem = None
        for stem in roughaligned_stems:
            if name == stem or name.startswith(stem + "."):
                if matched_stem is None or len(stem) > len(matched_stem):
                    matched_stem = stem
        if matched_stem is None:
            continue
        model = name[len(matched_stem) + 1 :]
        if not model:
            continue
        result.setdefault(matched_stem, {})[model] = p
    return result


# ---------------------------------------------------------------------------
# Per-file merging
# ---------------------------------------------------------------------------


def collect_error_occurrences(model_name: str, model_doc: dict, golden_data: dict[int, GoldenSegmentData]) -> list[dict]:
    """Flatten one model's output into a list of individual error occurrences,
    each tagged with its (golden_segment_index, word_range)."""
    occurrences = []
    for gseg_result in model_doc.get("golden_segments", []):
        gi = gseg_result["golden_segment_index"]
        gdata = golden_data.get(gi)
        if gdata is None:
            continue
        for chunk in gseg_result.get("chunks", []):
            if chunk["status"] != "issues found" or not chunk.get("errors"):
                continue
            chunk_range = tuple(chunk["golden_word_range"])
            for err in chunk["errors"]:
                word_range = locate_error_word_range(gdata.golden_words, chunk_range, err.get("source_words", ""))
                occurrences.append(
                    {
                        "model": model_name,
                        "golden_segment_index": gi,
                        "word_range": word_range,
                        "error_class": err.get("error_class", ""),
                        "source_words": err.get("source_words", ""),
                        "explanation": err.get("explanation", ""),
                        "renderings": err.get("renderings", []),
                    }
                )
    return occurrences


def cluster_occurrences(occurrences: list[dict]) -> list[list[dict]]:
    """Group occurrences (already scoped to one golden_segment_index) into
    clusters whose word ranges overlap or nearly touch (interval merge)."""
    by_gseg: dict[int, list[dict]] = {}
    for occ in occurrences:
        by_gseg.setdefault(occ["golden_segment_index"], []).append(occ)

    clusters: list[list[dict]] = []
    for gi, occs in by_gseg.items():
        occs = sorted(occs, key=lambda o: o["word_range"][0])
        current: list[dict] = []
        current_end = None
        for occ in occs:
            s, e = occ["word_range"]
            if current and s <= current_end + GAP_TOLERANCE_WORDS:
                current.append(occ)
                current_end = max(current_end, e)
            else:
                if current:
                    clusters.append(current)
                current = [occ]
                current_end = e
        if current:
            clusters.append(current)
    return clusters


def reconstruct_context(
    gdata: GoldenSegmentData, word_range: tuple[int, int]
) -> tuple[Optional[float], Optional[float], dict[str, str]]:
    """For a merged word range, reconstruct approximate start/end time
    (from any overlapping original automatic segment) and, per language,
    the full original automatic text overlapping that range."""
    start_time = end_time = None
    full_text: dict[str, str] = {}
    for lang, segs in gdata.lang_segments.items():
        matching = [s for s in segs if s["golden_word_range"] and ranges_overlap(s["golden_word_range"], word_range)]
        matching.sort(key=lambda s: s["automatic_segment_index"])
        full_text[lang] = " ".join(s["segment"] for s in matching)
        for s in matching:
            if start_time is None or s["start"] < start_time:
                start_time = s["start"]
            if end_time is None or s["end"] > end_time:
                end_time = s["end"]
    return start_time, end_time, full_text


def build_item(source_file: str, file_id: str, gdata: GoldenSegmentData, cluster: list[dict]) -> dict:
    starts = [o["word_range"][0] for o in cluster]
    ends = [o["word_range"][1] for o in cluster]
    word_range = (min(starts), max(ends))
    golden_transcript = " ".join(gdata.golden_words[word_range[0] - 1 : word_range[1]])
    start_time, end_time, full_text = reconstruct_context(gdata, word_range)

    models_reporting = sorted({o["model"] for o in cluster})
    explanations = [
        {
            "model": o["model"],
            "error_class": o["error_class"],
            "source_words": o["source_words"],
            "explanation": o["explanation"],
        }
        for o in cluster
    ]

    automatic_outputs = {}
    for lang in TARGET_AUTOMATIC_LANGS:
        if lang not in gdata.lang_segments:
            continue
        good, bad = [], []
        for o in cluster:
            for r in o["renderings"]:
                if r.get("language") != lang:
                    continue
                entry = {"model": o["model"], "text": r.get("text", "")}
                (good if r.get("correct") else bad).append(entry)
        automatic_outputs[lang] = {"full_text": full_text.get(lang, ""), "good": good, "bad": bad}

    return {
        "source_file": source_file,
        "id": file_id,
        "golden_segment_index": cluster[0]["golden_segment_index"],
        "golden_word_range": list(word_range),
        "start": start_time,
        "end": end_time,
        "golden_transcript": golden_transcript,
        "num_models_reporting": len(models_reporting),
        "models_reporting": models_reporting,
        "explanations": explanations,
        "automatic_outputs": automatic_outputs,
    }


def merge_file(
    roughaligned_path: Path, model_outputs: dict[str, Path]
) -> list[dict]:
    golden_data = load_roughaligned(roughaligned_path)
    file_id = roughaligned_path.stem.replace(".roughaligned", "")

    all_occurrences = []
    for model_name, out_path in model_outputs.items():
        try:
            model_doc = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read {out_path}: {e}", file=sys.stderr)
            continue
        all_occurrences.extend(collect_error_occurrences(model_name, model_doc, golden_data))

    items = []
    for cluster in cluster_occurrences(all_occurrences):
        gi = cluster[0]["golden_segment_index"]
        gdata = golden_data[gi]
        items.append(build_item(str(roughaligned_path), file_id, gdata, cluster))
    return items


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_json(items: list[dict], out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)


def write_csv(items: list[dict], out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_file",
                "id",
                "start",
                "end",
                "num_models_reporting",
                "models_reporting",
                "golden_transcript",
                "explanations",
                "en_bad",
                "de_bad",
                "cs_bad",
                "pl_bad",
                "sk_bad",
            ]
        )
        for it in items:
            explanations = " | ".join(
                f"[{e['model']}] {e['error_class']}: {e['source_words']} -- {e['explanation']}"
                for e in it["explanations"]
            )
            bad_by_lang = {
                lang: "; ".join(f"[{b['model']}] {b['text']}" for b in it["automatic_outputs"].get(lang, {}).get("bad", []))
                for lang in TARGET_AUTOMATIC_LANGS
            }
            writer.writerow(
                [
                    it["source_file"],
                    it["id"],
                    it["start"],
                    it["end"],
                    it["num_models_reporting"],
                    ",".join(it["models_reporting"]),
                    it["golden_transcript"],
                    explanations,
                    bad_by_lang.get("en", ""),
                    bad_by_lang.get("de", ""),
                    bad_by_lang.get("cs", ""),
                    bad_by_lang.get("pl", ""),
                    bad_by_lang.get("sk", ""),
                ]
            )
    tmp_path.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-model divergence findings into one reviewable list of error items (no LLM calls)."
    )
    parser.add_argument("--roughaligned-dir", default=str(SCRIPT_DIR / "roughaligned"))
    parser.add_argument("--divergencies-dir", default=str(SCRIPT_DIR / "02-divergencies-by-claude"))
    parser.add_argument("--output", default=str(SCRIPT_DIR / "merged_errors.json"))
    parser.add_argument("--csv-output", default=str(SCRIPT_DIR / "merged_errors.csv"))
    parser.add_argument(
        "--min-models",
        type=int,
        default=1,
        help="Only keep items reported by at least this many distinct models (default: 1, i.e. keep everything).",
    )
    args = parser.parse_args()

    roughaligned_dir = Path(args.roughaligned_dir).resolve()
    divergencies_dir = Path(args.divergencies_dir).resolve()

    roughaligned_paths = {p.stem: p for p in sorted(roughaligned_dir.glob("*.json"))}
    if not roughaligned_paths:
        print(f"ERROR: no roughaligned/*.json files found in {roughaligned_dir}", file=sys.stderr)
        sys.exit(1)

    stem_to_outputs = discover_model_outputs(divergencies_dir, set(roughaligned_paths))
    print(f"roughaligned files: {len(roughaligned_paths)}; with >=1 model output: {len(stem_to_outputs)}")

    all_items: list[dict] = []
    for i, (stem, roughaligned_path) in enumerate(sorted(roughaligned_paths.items()), 1):
        model_outputs = stem_to_outputs.get(stem)
        if not model_outputs:
            continue
        items = merge_file(roughaligned_path, model_outputs)
        all_items.extend(items)
        if i % 25 == 0 or i == len(roughaligned_paths):
            print(f"[{i}/{len(roughaligned_paths)}] {stem}: {len(items)} item(s) (models: {sorted(model_outputs)})", flush=True)

    if args.min_models > 1:
        all_items = [it for it in all_items if it["num_models_reporting"] >= args.min_models]

    all_items.sort(key=lambda it: (-it["num_models_reporting"], it["source_file"], it["golden_word_range"][0]))

    write_json(all_items, Path(args.output))
    write_csv(all_items, Path(args.csv_output))

    print("=" * 72)
    print(f"Done: {len(all_items)} merged error item(s) from {len(stem_to_outputs)} file(s).")
    print(f"JSON: {args.output}")
    print(f"CSV:  {args.csv_output}")


if __name__ == "__main__":
    main()
