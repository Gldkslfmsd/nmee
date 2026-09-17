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
    contributing model. Each "bad" rendering may additionally carry a
    "quote": a fine-grained substring of that rendering's own "text"
    pinpointing the actual erroneous word/phrase, extracted from the
    error's free-text "explanation" (which usually quotes the offending
    span per language, e.g. "Czech 'zpoždění' (delay) ...") and matched
    back against this rendering's text by string overlap. Falls back to
    no "quote" (whole rendering is the only thing to go on) when the
    explanation has no usable quote for that language.

Output: ./35-merged-annotations.jsonl, one JSON object per line in the
`annotations.jsonl` format documented in ../iwslt+claude-to-pearmut/README.md
(one object per merged error cluster, with a "targets" list holding one
entry per flagged language/rendering), plus ./35-merged-annotations.csv (flat
summary for quick spreadsheet triage). Fields beyond the README's schema
(source_file, golden_segment_index, golden_word_range, num_models_reporting,
models_reporting, automatic_texts) are kept as extra, non-required data --
the README explicitly allows this -- so merged_errors_to_pearmut.py can
rebuild the exact same Pearmut campaign it used to build from the old
merged_errors.json.

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
# Fine-grained quote extraction from explanations
#
# An error's "explanation" (written by the divergence-finding LLM) is a
# free-text sentence that usually quotes the actual offending word/phrase
# per language, e.g. "German 'Rückstand', Czech 'zpoždění' (delay) and
# Slovak 'nedostatky' ...". The per-language "bad" rendering we get from
# "renderings" is the whole quoted sentence/clause though, which is why
# highlighting used to bold the entire rendering instead of just the
# erroneous word/phrase. Here we pull every quoted span out of the
# explanation and, for each bad rendering, keep whichever one actually
# overlaps that rendering's own text -- i.e. the quote that belongs to that
# language, not some other language's quote that happens to sit nearby in
# the same sentence.
# ---------------------------------------------------------------------------

QUOTE_PATTERNS = [
    re.compile(r"'([^'\"‘’“”]{2,80})'"),   # straight single
    re.compile(r'"([^\'\"‘’“”]{2,80})"'),  # straight double
    re.compile(r"‘([^‘’]{2,80})’"),        # curly single
    re.compile(r"“([^“”]{2,80})”"),        # curly double
]


def extract_quotes(explanation: str) -> list[str]:
    """Pull every quoted substring out of an explanation string (any of the
    straight/curly quote styles seen in practice)."""
    quotes = []
    for pattern in QUOTE_PATTERNS:
        quotes.extend(m.group(1) for m in pattern.finditer(explanation))
    return quotes


def best_matching_quote(text: str, candidates: list[str]) -> Optional[str]:
    """Among quoted spans pulled from an explanation, return whichever one
    best overlaps `text` (one specific rendering), so a multi-language
    explanation's quotes get attributed to the right language. Requires the
    match to cover most of the candidate quote, so an unrelated quote (or an
    accidental match, e.g. an English contraction's apostrophe pair) isn't
    mistaken for a real one."""
    best, best_size = None, 0
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        if cand in text:
            size = len(cand)
        else:
            sm = difflib.SequenceMatcher(None, text, cand, autojunk=False)
            size = sm.find_longest_match(0, len(text), 0, len(cand)).size
        if size >= max(3, int(0.7 * len(cand))) and size > best_size:
            best, best_size = cand, size
    return best


def annotate_renderings_with_quotes(explanation: str, renderings: list[dict]) -> list[dict]:
    """Attach a "quote" field (the fine-grained erroneous span) to each
    incorrect rendering, when the explanation's quotes let us confidently
    identify one for that language."""
    quotes = extract_quotes(explanation)
    if not quotes:
        return renderings
    out = []
    for r in renderings:
        r2 = dict(r)
        if r.get("correct") is False:
            quote = best_matching_quote(r.get("text", ""), quotes)
            if quote is not None:
                r2["quote"] = quote
        out.append(r2)
    return out


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
    __slots__ = ("golden_words", "golden_word_times", "lang_segments")

    def __init__(
        self,
        golden_words: list[str],
        golden_word_times: list[dict],
        lang_segments: dict[str, list[dict]],
    ):
        self.golden_words = golden_words
        self.golden_word_times = golden_word_times
        self.lang_segments = lang_segments


def load_roughaligned(path: Path) -> dict[int, GoldenSegmentData]:
    """Map golden_segment_index -> GoldenSegmentData for one roughaligned file.

    golden_word_times are rough-align.py's real per-word timestamps for the
    golden transcript (anchored to the automatic English ASR's own word
    timestamps, see anchor_golden_word_times in rough-align.py), read
    straight from the roughaligned file, not recomputed here.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    available_langs = [l for l in TARGET_AUTOMATIC_LANGS if l in doc.get("languages", {}).get("automatic", [])]
    golden_word_times = doc.get("golden_reference", {}).get("words") or []
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
            golden_words=gseg["transcript"].split(),
            golden_word_times=golden_word_times,
            lang_segments=lang_segments,
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
                explanation = err.get("explanation", "")
                occurrences.append(
                    {
                        "model": model_name,
                        "golden_segment_index": gi,
                        "word_range": word_range,
                        "error_class": err.get("error_class", ""),
                        "source_words": err.get("source_words", ""),
                        "explanation": explanation,
                        "renderings": annotate_renderings_with_quotes(explanation, err.get("renderings", [])),
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
    """For a merged word range, reconstruct the start/end time and, per
    language, the full original automatic text overlapping that range.

    start/end come straight from the golden transcript's own per-word
    timestamps (golden_word_times, read from the roughaligned file's
    golden_reference.words -- see load_roughaligned), i.e. exactly the
    words in `word_range`, not from whichever automatic segment happens to
    be tagged as overlapping. This matters: an automatic segment's
    golden_word_range is only an estimate, so anchoring timing to it (as
    used to be done here) could pick up a wrong/neighboring segment's
    [start, end] whenever that estimate was slightly off. Golden's own
    timestamps are anchored directly to the automatic English ASR's
    real per-word timestamps (see anchor_golden_word_times in
    rough-align.py), so they stay correct even when a particular
    automatic segment's estimated golden_word_range is off.
    """
    start_time = end_time = None
    words = gdata.golden_word_times
    lo, hi = word_range
    if words and 1 <= lo <= len(words) and 1 <= hi <= len(words):
        start_time = words[lo - 1].get("start")
        end_time = words[hi - 1].get("end")

    full_text: dict[str, str] = {}
    for lang, segs in gdata.lang_segments.items():
        matching = [s for s in segs if s["golden_word_range"] and ranges_overlap(s["golden_word_range"], word_range)]
        matching.sort(key=lambda s: s["automatic_segment_index"])
        full_text[lang] = " ".join(s["segment"] for s in matching)
    return start_time, end_time, full_text


def find_span(text: str, needle: str) -> tuple[str, int, int, bool]:
    """Locate `needle` inside `text` (exact, case-insensitive, or best
    difflib match) and return (span, span_start, span_end, matched) such
    that `span == text[span_start:span_end]` always holds exactly --
    required by the annotations.jsonl target schema, which has no way to
    express "unknown location". When nothing usable is found, falls back to
    the whole text with matched=False, so callers that care (e.g. picking
    which error to visually highlight) can tell a real location from this
    fallback."""
    if not needle or not text:
        return text, 0, len(text), False
    idx = text.find(needle)
    if idx != -1:
        return text[idx : idx + len(needle)], idx, idx + len(needle), True
    idx = text.lower().find(needle.lower())
    if idx != -1:
        return text[idx : idx + len(needle)], idx, idx + len(needle), True
    sm = difflib.SequenceMatcher(None, text, needle, autojunk=False)
    match = sm.find_longest_match(0, len(text), 0, len(needle))
    if match.size >= max(4, len(needle) // 2):
        return text[match.a : match.a + match.size], match.a, match.a + match.size, True
    return text, 0, len(text), False


def build_lang_targets(lang: str, base_text: str, cluster: list[dict]) -> list[dict]:
    """Build the annotations.jsonl `targets[]` entries for one language of
    one merged cluster: one entry per (contributing model, bad rendering).
    All entries for the same (cluster, lang) share the same "system" and
    "text" (the pooled-longest rendering, `base_text`), per the README's
    "several errors in the same output are several entries with the same
    system and text" rule; only span/intended/explanation vary."""
    bad_pairs = []
    good_texts = []
    for o in cluster:
        for r in o["renderings"]:
            if r.get("language") != lang:
                continue
            if r.get("correct") is False:
                bad_pairs.append((o, r))
            elif r.get("correct"):
                good_texts.append(r.get("text", ""))
    if not bad_pairs:
        return []

    # Best-effort "intended" (what the span should have said): the longest
    # rendering some other model judged *correct* for this same language and
    # cluster, if any -- otherwise fall back to the original English words
    # the error was anchored to, which is the closest thing we have.
    intended_fallback = max(good_texts, key=len) if good_texts else None

    system = "asr" if lang == "en" else "asr+mt"
    error_source = "ASR" if lang == "en" else "MT"

    targets = []
    for o, r in bad_pairs:
        span, span_start, span_end, span_matched = find_span(base_text, r.get("text", ""))
        error_class = o.get("error_class", "")
        explanation = o.get("explanation", "")
        targets.append(
            {
                "tgt_lan": lang,
                "system": system,
                "text": base_text,
                "reference": None,
                "span": span,
                "span_start": span_start,
                "span_end": span_end,
                "intended": intended_fallback if intended_fallback is not None else o.get("source_words", ""),
                # We only have the divergence-finder's free-text error class,
                # not a genuine harm judgement, so we can't confidently sort
                # errors into the README's harm categories -- default to
                # "Other" and keep the real class in `explanation`.
                "harm_types": ["Other"],
                # A "quote" was only attached when the explanation's own
                # wording could be confidently matched to this rendering (see
                # annotate_renderings_with_quotes); treat that as a stronger
                # signal than a bare whole-rendering match.
                "confidence": "harmful" if r.get("quote") else "borderline",
                "borderline": not bool(r.get("quote")),
                "explanation": f"{error_class}: {explanation}" if error_class else explanation,
                "error_source": error_source,
                # Extra, non-README bookkeeping: which divergence-finder LLM
                # reported this, needed by merged_errors_to_pearmut.py to
                # reproduce the old per-language "how many distinct models
                # flagged this" ranking; and whether `span` is a real located
                # match or just the find_span() whole-text fallback (needed
                # to reproduce the old "don't highlight anything when we
                # can't confidently locate the error" behavior).
                "annotator_model": o["model"],
                "span_matched": span_matched,
            }
        )
    return targets


def build_item(source_file: str, file_id: str, gdata: GoldenSegmentData, cluster: list[dict]) -> dict:
    starts = [o["word_range"][0] for o in cluster]
    ends = [o["word_range"][1] for o in cluster]
    word_range = (min(starts), max(ends))
    golden_transcript = " ".join(gdata.golden_words[word_range[0] - 1 : word_range[1]])
    start_time, end_time, full_text = reconstruct_context(gdata, word_range)

    models_reporting = sorted({o["model"] for o in cluster})

    automatic_texts: dict[str, str] = {}
    targets: list[dict] = []
    for lang in TARGET_AUTOMATIC_LANGS:
        if lang not in gdata.lang_segments:
            continue
        good_texts, bad_texts = [], []
        for o in cluster:
            for r in o["renderings"]:
                if r.get("language") != lang:
                    continue
                (good_texts if r.get("correct") else bad_texts).append(r.get("text", ""))
        pool = good_texts + bad_texts
        base_text = max(pool, key=len) if pool else full_text.get(lang, "")
        automatic_texts[lang] = base_text
        targets.extend(build_lang_targets(lang, base_text, cluster))

    gi = cluster[0]["golden_segment_index"]
    doc_id = f"{file_id}__gi{gi}__w{word_range[0]}-{word_range[1]}"

    return {
        # --- annotations.jsonl schema fields (see README.md) ---
        "doc_id": doc_id,
        "filename": f"{file_id}.mp3",
        "segment": gi,
        "orig_start": start_time,
        "orig_end": end_time,
        "gold_transcript": golden_transcript,
        "asr": automatic_texts.get("en"),
        "targets": targets,
        # --- extra fields (README: "Other fields are optional") ---
        "source_file": source_file,
        "id": file_id,
        "golden_segment_index": gi,
        "golden_word_range": list(word_range),
        "num_models_reporting": len(models_reporting),
        "models_reporting": models_reporting,
        "automatic_texts": automatic_texts,
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
        item = build_item(str(roughaligned_path), file_id, gdata, cluster)
        if not item["targets"]:
            # Every cluster originates from a reported error, but if none of
            # its renderings were confidently marked "incorrect" for any
            # language, there is nothing to put in targets[] -- skip.
            continue
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_jsonl(items: list[dict], out_path: Path) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")
    tmp_path.replace(out_path)


def write_csv(items: list[dict], out_path: Path) -> None:
    """Flat summary, one row per target (per flagged language/rendering)."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "doc_id",
                "filename",
                "orig_start",
                "orig_end",
                "num_models_reporting",
                "models_reporting",
                "gold_transcript",
                "tgt_lan",
                "system",
                "span",
                "intended",
                "confidence",
                "error_source",
                "explanation",
            ]
        )
        for it in items:
            for t in it["targets"]:
                writer.writerow(
                    [
                        it["doc_id"],
                        it["filename"],
                        it["orig_start"],
                        it["orig_end"],
                        it["num_models_reporting"],
                        ",".join(it["models_reporting"]),
                        it["gold_transcript"],
                        t["tgt_lan"],
                        t["system"],
                        t["span"],
                        t["intended"],
                        t["confidence"],
                        t["error_source"],
                        t["explanation"],
                    ]
                )
    tmp_path.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-model divergence findings into one reviewable list of error items (no LLM calls)."
    )
    parser.add_argument("--roughaligned-dir", default=str(SCRIPT_DIR / "roughaligned"))
    parser.add_argument("--divergencies-dir", default=str(SCRIPT_DIR / "02-divergencies-by-claude"))
    parser.add_argument("--output", default=str(SCRIPT_DIR / "35-merged-annotations.jsonl"))
    parser.add_argument("--csv-output", default=str(SCRIPT_DIR / "35-merged-annotations.csv"))
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

    write_jsonl(all_items, Path(args.output))
    write_csv(all_items, Path(args.csv_output))

    print("=" * 72)
    print(f"Done: {len(all_items)} merged error item(s) from {len(stem_to_outputs)} file(s).")
    print(f"JSONL: {args.output}")
    print(f"CSV:   {args.csv_output}")


if __name__ == "__main__":
    main()
