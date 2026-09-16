#!/usr/bin/env python3
"""Find laughable/erroneous translation & transcription divergencies in
./15-roughaligned/*.json using an LLM via DSPy.

Each ./15-roughaligned/*.json file (produced by rough-align.py) holds, per
input audio portion, one or more golden English segments together with,
for every target-side language ("automatic": en/de/cs/pl/sk -- 'en' being
the automatic ASR transcript of the English speech, the other four being
automatic translations), a list of automatic segments that overlap it in
time, each already tagged with the 1-based inclusive [golden_word_range] of
golden-transcript words it corresponds to (see rough-align.py). In this
dataset there happens to be exactly one golden segment per file (spanning
the whole ~600s portion), so the unit of analysis below is simply that
golden segment.

For each golden segment, the LLM is given the FULL golden transcript and,
for each language, ALL of that language's overlapping automatic segments
concatenated in chronological order (this is exactly what the hardcoded
prompt below describes: individual automatic segments cover only a
portion, but concatenated per language they cover the whole golden
transcript, possibly with a few extra words at the edges). Giving the LLM
this full context -- rather than a narrow, pre-cut slice -- lets it place
chunk boundaries at real sentence/clause boundaries and correctly judge
what is genuinely missing/mistranslated vs. what merely belongs to a
neighboring chunk, instead of flagging pure slicing artifacts.

The LLM walks the golden transcript from start to end and emits one entry
("chunk") per natural sub-segment, each either "all outputs correct" or
carrying a list of divergence occurrences. It does NOT need to (and is not
asked to) invent timestamps or original-segment identities -- the calling
script reconstructs, for every returned chunk, which of the *original*
per-language automatic segments (with their real automatic_segment_index,
start and end, already known from roughaligned/*.json) it corresponds to,
by locating the chunk's quoted golden-transcript text within the full
golden transcript and looking up which original segments' own
golden_word_range overlaps it.

Output: ./25-divergencies-by-llms/<input-stem>.<model>.json

Usage
-----
    ./venv/bin/python 20-find-bad-translation-divergencies.py --model=gpt-oss-120B 15-roughaligned/423057182_1193.80_1790.84.roughaligned.json
    ./venv/bin/python 20-find-bad-translation-divergencies.py --model=GLM 15-roughaligned/*.json
"""

import argparse
import concurrent.futures
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# DSPy must have its cache directory pinned *before* it is imported: DSPy's
# diskcache (SQLite/FanoutCache) misbehaves on network filesystems such as
# /lnet -- WAL journal mode mmaps a -shm file (unreliable over NFS, causes
# SIGBUS) and its default 16 shards each cost an fsync round-trip on open
# (~2.5s/shard -> ~40s startup). Fix both before `import dspy` creates the
# cache object. See ./inspiration-dspy-classifier/classify_raw_strings.py.
# ---------------------------------------------------------------------------

if "DSPY_CACHEDIR" not in os.environ:
    os.environ["DSPY_CACHEDIR"] = str(SCRIPT_DIR / "dspy_cache")

import diskcache.core  # noqa: E402

diskcache.core.DEFAULT_SETTINGS["sqlite_journal_mode"] = "delete"
diskcache.core.DEFAULT_SETTINGS["sqlite_mmap_size"] = 0

import diskcache.fanout  # noqa: E402

_OrigFanoutCache = diskcache.fanout.FanoutCache


class _SingleShardFanoutCache(_OrigFanoutCache):
    """FanoutCache that always uses 1 shard regardless of caller's request."""

    def __init__(self, directory=None, shards=8, timeout=0.010, disk=diskcache.core.Disk, **settings):
        super().__init__(directory=directory, shards=1, timeout=timeout, disk=disk, **settings)


diskcache.fanout.FanoutCache = _SingleShardFanoutCache
diskcache.FanoutCache = _SingleShardFanoutCache

import dspy  # noqa: E402
from dspy.utils.exceptions import is_retryable_lm_error  # noqa: E402

# ---------------------------------------------------------------------------
# e-INFRA model registry.
#
# The keys below are exactly the MODEL names used on the command line and in
# output filenames (25-divergencies-by-llms/<input-stem>.MODEL.json).
# ---------------------------------------------------------------------------

_EINFRA_URL = "https://llm.ai.e-infra.cz/v1/"
_EINFRA_KEYS = ["E_INFRA_API_TOKEN", "CESNET_API_KEY"]

# Some e-INFRA models are "thinking" models that emit a hidden
# reasoning_content channel before their final answer. On our large,
# structured-output FindDivergencies call, that reasoning can consume the
# ENTIRE max_tokens budget, leaving zero room for the actual JSON answer
# (observed: GLM/Kimi/DeepSeek all returned an empty final response at
# max_tokens=16000). Passing reasoning_effort="low" (via litellm's
# allowed_openai_params escape hatch, since litellm's generic "openai/"
# provider otherwise rejects it as an unsupported param) fixes this for
# GLM and Kimi; DeepSeek needs it too but remains much slower regardless
# (large real calls observed to take 270-500s+, vs. gpt-oss-120B's
# 30-90s), so it also gets a longer --timeout.
_LOW_REASONING_KWARGS = {"allowed_openai_params": ["reasoning_effort"], "reasoning_effort": "low"}

KNOWN_MODELS: dict[str, dict] = {
    "Gemma4": {
        "api_model": "openai/gemma4",
        "base_url": _EINFRA_URL,
        "env_var": _EINFRA_KEYS,
        "extra_lm_kwargs": {},
    },
    "gpt-oss-120B": {
        "api_model": "openai/gpt-oss-120b",
        "base_url": _EINFRA_URL,
        "env_var": _EINFRA_KEYS,
        "extra_lm_kwargs": {},
    },
    "qwen3.8-27b": {
        "api_model": "openai/qwen3.8-27b",
        "base_url": _EINFRA_URL,
        "env_var": _EINFRA_KEYS,
        "extra_lm_kwargs": {},
    },
    "GLM": {
        "api_model": "openai/glm",
        "base_url": _EINFRA_URL,
        "env_var": _EINFRA_KEYS,
        "extra_lm_kwargs": _LOW_REASONING_KWARGS,
    },
    "Kimi": {
        "api_model": "openai/kimi",
        "base_url": _EINFRA_URL,
        "env_var": _EINFRA_KEYS,
        "extra_lm_kwargs": _LOW_REASONING_KWARGS,
    },
    "DeepSeek": {
        "api_model": "openai/deepseek-v4-flash",
        "base_url": _EINFRA_URL,
        "env_var": _EINFRA_KEYS,
        "extra_lm_kwargs": _LOW_REASONING_KWARGS,
    },
}

DEFAULT_MODEL = "gpt-oss-120B"


def _resolve_api_key(env_var_names: list[str]) -> str:
    for name in env_var_names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    names = ", ".join(env_var_names)
    print(
        f"ERROR: None of the API-key environment variables are set: {names}\n"
        f"Please export one of them before running this script.",
        file=sys.stderr,
    )
    sys.exit(1)


def get_lm(modelname: str, **kwargs) -> dspy.LM:
    if modelname not in KNOWN_MODELS:
        available = ", ".join(sorted(KNOWN_MODELS))
        print(f"ERROR: Unknown model {modelname!r}.\nAvailable models: {available}", file=sys.stderr)
        sys.exit(1)
    entry = KNOWN_MODELS[modelname]
    api_key = _resolve_api_key(entry["env_var"])
    merged_kwargs = {**entry.get("extra_lm_kwargs", {}), **kwargs}
    return dspy.LM(entry["api_model"], base_url=entry["base_url"], api_key=api_key, **merged_kwargs)


def configure_lm(modelname: str, **kwargs) -> dspy.LM:
    lm = get_lm(modelname, **kwargs)
    dspy.configure(lm=lm)
    return lm


# ---------------------------------------------------------------------------
# DSPy signature & module
#
# The HARDCODED PROMPT (verbatim, as specified) is the docstring of
# FindDivergencies below.
# ---------------------------------------------------------------------------


class LanguageOutput(BaseModel):
    """One language's full automatic output for this golden segment (all of
    that language's overlapping automatic segments, concatenated in
    chronological order)."""

    language: str = Field(
        description="Language code: 'en' for the automatic ASR transcript of the English speech, "
        "or 'de'/'cs'/'pl'/'sk' for the automatic translation."
    )
    text: str = Field(description="The full concatenated automatic transcription/translation text.")


class LanguageRendering(BaseModel):
    """One language's rendering of a specific error occurrence, marked good or bad."""

    language: str = Field(description="Language code: 'en', 'de', 'cs', 'pl', or 'sk'.")
    text: str = Field(
        description="The exact word(s)/expression in this language corresponding to source_words for this occurrence."
    )
    correct: bool = Field(
        description="True if this language's rendering correctly preserves the meaning of source_words; "
        "False if this is the erroneous/divergent rendering."
    )


class ErrorOccurrence(BaseModel):
    """One instance of a laughable translation/transcription divergence within the chunk."""

    error_class: str = Field(
        description="Short label for the error type, e.g. 'domain/genre mismatch', 'bad segmentation', "
        "'mistranslated proper name', 'badly rendered/conjugated name', 'gender mismatch', or similar."
    )
    source_words: str = Field(description="The English source words or expression this occurrence concerns.")
    explanation: str = Field(description="Brief explanation of why this is an error.")
    renderings: list[LanguageRendering] = Field(
        description="One entry per available language (en, de, cs, pl, sk), each recording its text and "
        "whether it is correct or the erroneous rendering for this occurrence."
    )


class ChunkResult(BaseModel):
    """One natural contiguous chunk of the golden transcript and its verdict."""

    transcript: str = Field(
        description="The verbatim (or as-close-to-verbatim-as-possible) span of golden_transcript this chunk covers."
    )
    status: Literal["all outputs correct", "issues found"] = Field(
        description="'all outputs correct' if every automatic output faithfully preserves the meaning of this "
        "chunk; 'issues found' if at least one divergence occurrence was identified in it."
    )
    errors: list[ErrorOccurrence] = Field(
        default_factory=list,
        description="All divergence occurrences found in this chunk (empty list if status is 'all outputs correct').",
    )


class FindDivergencies(dspy.Signature):
    """You are a very culturally sensitive and knowledgeable professional translator, also experienced in
consecutive and simultaneous interpreting. You are provided with a short English source text (golden truth
transcription of a speech) and a set of roughly corresponding automatic transcription of the English speech
and automatic translations of the English speech into German, Czech, Polish and Slovak. Individually, these
segments may cover only a portion of the input English golden transcript. Concatenated in each language, they
cover all the English golden transcript and may contain a few extra words at the start or end.

Most of the segments will be transcribed and translated correctly. For these cases, emit an entry stating
golden_segment_index, start, end, transcript and status "all outputs correct".

Identify segments where any of the languages differs in the meaning. We are in particular searching for
laughable translation or transcription differences: words or expressions that are not preserved but replaced
with an expression from a totally different domain, genre, or settings; words that were badly segmented and
transcribed or translated by unrelated parts; any names that got badly translated or simply badly rendered
(e.g. conjugated) in the target language; detectable mismatches in gender of people discussed or referred to
(as soon there is any divergence in gender across the languages, that is an indication of the error). For each
such occurrence within an input segment, record error class (the one of the above or similar), the source
words and the corresponding transcriptions and translations in each of the languages. The output entry should
record all the outputs, distinguishing the good ones and those which have the given error type.

There can be more error occurrences in a given input segment, record all."""

    golden_segment_index: int = dspy.InputField(description="Index of the golden segment being analyzed.")
    golden_transcript: str = dspy.InputField(description="The full golden truth English transcript for this segment.")
    automatic_outputs: list[LanguageOutput] = dspy.InputField(
        description="For each language (en=automatic ASR transcript of the English speech, de/cs/pl/sk=automatic "
        "translations), ALL of that language's segments concatenated in chronological order -- together they "
        "cover the whole golden_transcript, possibly with a few extra words at the very start or end."
    )
    chunks: list[ChunkResult] = dspy.OutputField(
        description="Walk through golden_transcript from start to end, in order, splitting it into natural "
        "contiguous chunks (e.g. sentences or clauses) that together cover the ENTIRE golden_transcript with no "
        "gaps and no overlaps. Emit exactly one entry per chunk, quoting the golden_transcript words it covers "
        "as closely to verbatim as possible."
    )


class DivergenceFinder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(FindDivergencies)

    def forward(self, golden_segment_index: int, golden_transcript: str, automatic_outputs: list):
        return self.predict(
            golden_segment_index=golden_segment_index,
            golden_transcript=golden_transcript,
            automatic_outputs=automatic_outputs,
        )


# ---------------------------------------------------------------------------
# Input file handling: build one analysis unit per golden segment (full
# golden transcript + full per-language concatenation), keeping the original
# per-language segment metadata around for reconstruction after the LLM call.
# ---------------------------------------------------------------------------

TARGET_AUTOMATIC_LANGS = ["en", "de", "cs", "pl", "sk"]


class AnalysisUnit:
    __slots__ = ("golden_segment_index", "golden_segment", "automatic_outputs", "lang_segment_meta")

    def __init__(self, golden_segment_index, golden_segment, automatic_outputs, lang_segment_meta):
        self.golden_segment_index = golden_segment_index
        self.golden_segment = golden_segment
        self.automatic_outputs = automatic_outputs
        self.lang_segment_meta = lang_segment_meta


def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def build_analysis_units(doc: dict) -> list[AnalysisUnit]:
    available_langs = [l for l in TARGET_AUTOMATIC_LANGS if l in doc.get("languages", {}).get("automatic", [])]
    units: list[AnalysisUnit] = []

    for gseg in doc.get("segments", []):
        overlapping = gseg.get("automatic_overlapping_segments", {})
        automatic_outputs = []
        lang_segment_meta: dict[str, list[dict]] = {}

        for lang in available_langs:
            lang_segments = sorted(overlapping.get(lang, []), key=lambda s: s["automatic_segment_index"])
            automatic_outputs.append(LanguageOutput(language=lang, text=" ".join(s["segment"] for s in lang_segments)))
            lang_segment_meta[lang] = [
                {
                    "automatic_segment_index": s["automatic_segment_index"],
                    "start": s["start"],
                    "end": s["end"],
                    "golden_word_range": tuple(s["golden_word_range"]) if s.get("golden_word_range") else None,
                }
                for s in lang_segments
            ]

        units.append(
            AnalysisUnit(
                golden_segment_index=gseg["golden_segment_index"],
                golden_segment=gseg,
                automatic_outputs=automatic_outputs,
                lang_segment_meta=lang_segment_meta,
            )
        )
    return units


# ---------------------------------------------------------------------------
# Reconstruction: map each LLM-returned chunk back onto the original,
# already-known per-language automatic segments (index/start/end), by
# locating the chunk's quoted golden-transcript text within the full golden
# transcript and checking golden_word_range overlap -- no timestamps or
# segment identities are trusted from the LLM itself.
# ---------------------------------------------------------------------------


def align_words(golden_words: list[str], pos: int, chunk_words: list[str], slack: int = 20) -> tuple[int, int]:
    """Best-effort, forward-only: locate chunk_words within golden_words at
    or after `pos`. Returns a 0-based half-open (start, end) range, always
    making forward progress so reconstruction can never stall."""
    if not chunk_words:
        return pos, pos + 1
    window_end = min(len(golden_words), pos + len(chunk_words) + slack)
    window = golden_words[pos:window_end]
    sm = difflib.SequenceMatcher(None, window, chunk_words, autojunk=False)
    match = sm.find_longest_match(0, len(window), 0, len(chunk_words))
    if match.size == 0:
        start_idx = pos
    else:
        start_idx = pos + max(0, match.a - match.b)
    end_idx = min(max(start_idx + len(chunk_words), start_idx + 1), len(golden_words))
    return start_idx, end_idx


def reconstruct_chunks(unit: AnalysisUnit, chunks: list[ChunkResult]) -> list[dict]:
    golden_words = unit.golden_segment["transcript"].split()
    pos = 0
    out = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_words = chunk.transcript.split()
        start_idx, end_idx = align_words(golden_words, pos, chunk_words)
        golden_word_range = (start_idx + 1, end_idx)  # 1-based inclusive
        pos = max(end_idx, pos + 1)

        reconstructed_segments: dict[str, list[int]] = {}
        chunk_start = chunk_end = None
        for lang, segs in unit.lang_segment_meta.items():
            matching = [s for s in segs if s["golden_word_range"] and ranges_overlap(s["golden_word_range"], golden_word_range)]
            reconstructed_segments[lang] = [s["automatic_segment_index"] for s in matching]
            for s in matching:
                if chunk_start is None or s["start"] < chunk_start:
                    chunk_start = s["start"]
                if chunk_end is None or s["end"] > chunk_end:
                    chunk_end = s["end"]

        out.append(
            {
                "golden_segment_index": unit.golden_segment_index,
                "chunk_index": chunk_index,
                "golden_word_range": list(golden_word_range),
                "start": chunk_start,
                "end": chunk_end,
                "transcript": chunk.transcript,
                "status": chunk.status,
                "reconstructed_segments": reconstructed_segments,
                "errors": [e.model_dump() for e in chunk.errors],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Transient-error backoff (same policy as classify_raw_strings.py)
# ---------------------------------------------------------------------------

_RESET_AT_RE = re.compile(r"resets at:\s*([\d-]+\s[\d:]+)\s*UTC", re.IGNORECASE)


def transient_error_delay(exc: Exception, consecutive_hits: int, base_delay: float, max_delay: float) -> float:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        return min(float(retry_after) + 0.5, max_delay)
    m = _RESET_AT_RE.search(str(exc))
    if m:
        try:
            reset_dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            delay = (reset_dt - datetime.now(timezone.utc)).total_seconds()
            if delay > 0:
                return min(delay + 0.5, max_delay)
        except ValueError:
            pass
    return min(base_delay * (2 ** (consecutive_hits - 1)), max_delay)


def analyze_unit(
    finder: DivergenceFinder,
    unit: AnalysisUnit,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> dict:
    """Run one golden segment through the LLM (with retries on transient
    errors) and reconstruct its chunks. On persistent failure, returns a
    single ERROR placeholder entry instead of raising."""
    attempt = 0
    consecutive_hits = 0
    while True:
        attempt += 1
        try:
            result = finder(
                golden_segment_index=unit.golden_segment_index,
                golden_transcript=unit.golden_segment["transcript"],
                automatic_outputs=unit.automatic_outputs,
            )
            return {
                "golden_segment_index": unit.golden_segment_index,
                "status": "ok",
                "chunks": reconstruct_chunks(unit, list(result.chunks)),
            }
        except Exception as e:
            if is_retryable_lm_error(e) and attempt <= max_retries:
                consecutive_hits += 1
                delay = transient_error_delay(e, consecutive_hits, base_delay, max_delay)
                time.sleep(delay)
                continue
            return {
                "golden_segment_index": unit.golden_segment_index,
                "status": "ERROR",
                "error_message": repr(e),
                "chunks": [],
            }


def process_file(
    in_path: Path,
    out_path: Path,
    model_name: str,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> dict:
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    units = build_analysis_units(doc)
    finder = DivergenceFinder()

    golden_segment_results = [analyze_unit(finder, unit, max_retries, base_delay, max_delay) for unit in units]
    all_chunks = [c for r in golden_segment_results for c in r["chunks"]]

    out_doc = {
        "id": doc.get("id"),
        "source_file": str(in_path),
        "model": model_name,
        "num_golden_segments": len(units),
        "num_golden_segment_errors": sum(1 for r in golden_segment_results if r["status"] == "ERROR"),
        "num_chunks": len(all_chunks),
        "num_chunks_with_issues": sum(1 for c in all_chunks if c["status"] == "issues found"),
        "golden_segments": golden_segment_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    tmp_path.replace(out_path)
    return out_doc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find laughable translation/transcription divergencies in roughaligned/*.json via an LLM."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more ./roughaligned/*.json files to process (shell-glob-expanded, e.g. roughaligned/*.json).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to use (default: {DEFAULT_MODEL}). Available: {', '.join(sorted(KNOWN_MODELS))}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "25-divergencies-by-llms"),
        help="Directory to write <input-stem>.<model>.json into (default: ./25-divergencies-by-llms).",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="LM sampling temperature (default: 0.0).")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of FILES to process in parallel (default: 4 -- the e-INFRA gateway caps in-flight requests "
        "per API key at 4; each file normally issues one LLM call per golden segment).",
    )
    parser.add_argument("--num-retries", type=int, default=1, help="LM-level retries on transient errors (default: 1).")
    parser.add_argument(
        "--our-max-retries",
        type=int,
        default=5,
        help="Our own retry attempts on transient errors before giving up on a golden segment (default: 5).",
    )
    parser.add_argument("--transient-base-delay", type=float, default=2.0, help="Base backoff delay in seconds (default: 2.0).")
    parser.add_argument("--transient-max-delay", type=float, default=60.0, help="Max backoff delay in seconds (default: 60.0).")
    parser.add_argument(
        "--timeout",
        type=float,
        default=450.0,
        help="Per-request timeout in seconds (default: 450) -- one call now covers a whole golden segment; "
        "reasoning-heavy models (Kimi, DeepSeek) have been observed taking 270-400s+ on real files.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
        help="LM max output tokens (default: 16000) -- one call now covers the WHOLE golden segment (dozens of "
        "chunks), so this needs to be generous; reasoning models also spend part of it on hidden thinking tokens.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess even if the output file already exists (default: skip files already done).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    lm = configure_lm(
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache=True,
        num_retries=args.num_retries,
        timeout=args.timeout,
    )
    print(f"Model: {args.model}  ({lm.model})")
    print(f"Output dir: {output_dir}")
    print(f"Concurrency (files): {args.concurrency}")
    print(flush=True)

    in_paths = [Path(p) for p in args.inputs]
    todo = []
    for in_path in in_paths:
        out_path = output_dir / f"{in_path.stem}.{args.model}.json"
        if out_path.exists() and not args.overwrite:
            print(f"SKIP {in_path.name} (output exists: {out_path.name})", flush=True)
            continue
        todo.append((in_path, out_path))

    total_chunks = 0
    total_issues = 0
    workers = max(1, min(args.concurrency, len(todo))) if todo else 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_file,
                in_path,
                out_path,
                args.model,
                args.our_max_retries,
                args.transient_base_delay,
                args.transient_max_delay,
            ): in_path
            for in_path, out_path in todo
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            in_path = futures[future]
            try:
                out_doc = future.result()
                total_chunks += out_doc["num_chunks"]
                total_issues += out_doc["num_chunks_with_issues"]
                print(
                    f"[{i}/{len(todo)}] {in_path.name}: done "
                    f"({out_doc['num_chunks']} chunk(s), {out_doc['num_chunks_with_issues']} with issues, "
                    f"{out_doc['num_golden_segment_errors']} golden-segment error(s))",
                    flush=True,
                )
            except Exception as e:
                print(f"[{i}/{len(todo)}] {in_path.name}: FAILED: {e}", file=sys.stderr, flush=True)

    print("=" * 72)
    print(f"Done: {len(todo)} file(s) processed, {total_chunks} chunk(s) total, {total_issues} with issues found.")


if __name__ == "__main__":
    main()
