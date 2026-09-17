#!/usr/bin/env python3
"""Convert ./35-merged-annotations.jsonl (the annotations.jsonl format
documented in ../iwslt+claude-to-pearmut/README.md, one JSON object per
line) into a Pearmut (github.com/zouharvi/pearmut) annotation-campaign JSON
file, so humans can validate the translation errors found automatically by
the LLM-based divergence finder.

For each merged-annotations.jsonl line we build one Pearmut item:
  - a short audio clip (cut out of the long source_file mp3 with ffmpeg,
    using the item's "start"/"end" offsets) plus the golden_transcript text,
    shown as the source side;
  - up to 3 target-language renderings, chosen from {en, de, cs, sk, pl}
    (always displayed left-to-right in that order):
      * "en" is included whenever any model flagged an error in the English
        transcript itself.
      * the remaining slots (2 if en is shown, else 3) go to the languages
        among {de, cs, sk, pl} with the most distinct models reporting an
        error there (ties broken by the de/cs/sk/pl priority order).
    For each shown language we display the single longest quoted rendering
    across all contributing models' "good"/"bad" judgements for that
    language, and bold (<b>) the longest "bad" (erroneous) quote within it,
    so annotators can spot the flagged span quickly.

Items are grouped --items-per-screen (default 10) per screen (document), and
--screens-per-link (default 20) screens per annotator link ("task-based"
Pearmut pages/tasks) -- 200 items/link by default.

Cut audio clips are written to ./data/assets/<id>_<newstart>_<newend>.mp3
and referenced as /assets/<...>.mp3, because Pearmut's server serves static
assets from <PEARMUT_ROOT>/data/assets/ at the /assets/ URL path (PEARMUT_ROOT
defaults to "."). Run `pearmut run` from this project directory so that its
default data/ root lines up with the audio already placed here.

Needs only the Python standard library, plus the `ffmpeg` binary on PATH
(present at /usr/bin/ffmpeg in this environment) to cut audio clips.

Usage
-----
    ./venv/bin/python merged_errors_to_pearmut.py
    ./venv/bin/python merged_errors_to_pearmut.py --limit 100  # quick test run
"""

# NOTE on reconstructing the old per-language picks from the new schema:
# 30-merge-annotations.py's build_item() stores, per language, the pooled
# "longest good+bad rendering" text as both `targets[].text` (for every
# target of that language) and, redundantly, under the extra `automatic_texts`
# field (also covering languages with no error at all, needed to fill
# language slots when too few languages have errors) -- this lets us pick
# the same per-language display text and the same "how many distinct models
# flagged this language" ranking as the old merged_errors.json-based code did.

import argparse
import html
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

LANG_DISPLAY_ORDER = ["en", "de", "cs", "sk", "pl"]
NON_EN_LANGS = ["de", "cs", "sk", "pl"]
NON_EN_PRIORITY = {lang: i for i, lang in enumerate(NON_EN_LANGS)}

# Flagged word spans are now precisely time-anchored (see rough-align.py's
# anchor_golden_word_times), so a raw span is often under a second -- too
# short to give an annotator any spoken context. Pad the audio clip (not
# merged_errors.json's own start/end, which stay exact) by this many
# seconds on each side, clamped to the recording's bounds.
CONTEXT_PAD_SECONDS = 2.0

PROTOCOL_INSTRUCTIONS_ESA = """
<p><b>Task:</b> Listen to the short source-audio clip (and read its transcript),
then check the translation(s) shown below it, one per language. A
<b>bold</b> span marks text that our automatic error-finding pipeline
flagged as a possible translation error &mdash; please verify whether it
really is one.</p>
<ul>
  <li>Error spans:
    <ul>
      <li><b>Click</b> on the start of an error, then <b>click</b> on the end to mark an
        error span.</li>
      <li><b>Hover</b> over an existing highlight to change error severity (minor/major) or remove it.
      </li>
    </ul>
    Error severity:
    <ul>
      <li><span class="error_minor">Minor:</span> Style, grammar, or word choice
        could be better.</li>
      <li><span class="error_major">Major:</span> Meaning is significantly
        changed or is hard to understand.</li>
    </ul>
    <b>Tip</b>: Mark the general area of the error (doesn't need to be exact). Use separate highlights
    for different errors.
    Use <code style="font-family: monospace">[missing]</code> at the end of a sentence for omitted content.<br>
  </li>
  <li>Score each translation using the slider based on meaning preservation and quality.
    <b>Important:</b> The relative order of scores matters; ensure better translations have higher
    scores than worse ones.
    <ul>
      <li>0: <b>Broken</b>/Nonsense</li>
      <li>33%: <b>Flawed</b>: substantial issues.</li>
      <li>66%: <b>Good</b>: small issues with grammar, fluency, or consistency.</li>
      <li>100%: <b>Perfect</b>: meaning and style align completely with the source.</li>
    </ul>
  </li>
</ul>
"""


def parse_id(item_id: str) -> tuple[str, float, float]:
    """"<numeric_id>_<long_segment_start>_<long_segment_end>" -> parts."""
    numeric_id, long_start_s, long_end_s = item_id.split("_")
    return numeric_id, float(long_start_s), float(long_end_s)


def is_valid_span(it: dict) -> bool:
    start, end = it["orig_start"], it["orig_end"]
    if start is None or end is None:
        return False
    _, long_start, long_end = parse_id(it["id"])
    duration = long_end - long_start
    if start < 0 or end <= start or end > duration + 1.0:
        return False
    return True


def padded_span(it: dict) -> tuple[float, float]:
    """The item's exact [orig_start, orig_end] widened by
    CONTEXT_PAD_SECONDS on each side (clamped to the recording's own
    duration) -- used only for cutting a listenable audio clip, not for
    merged-annotations.jsonl's own orig_start/orig_end."""
    _, long_start, long_end = parse_id(it["id"])
    duration = long_end - long_start
    pad_start = max(0.0, it["orig_start"] - CONTEXT_PAD_SECONDS)
    pad_end = min(duration, it["orig_end"] + CONTEXT_PAD_SECONDS)
    return pad_start, pad_end


# ---------------------------------------------------------------------------
# Audio cutting
# ---------------------------------------------------------------------------


def cut_audio(src_path: Path, start: float, end: float, out_path: Path) -> str | None:
    """Cut [start, end) out of src_path into out_path with ffmpeg. Returns an
    error string on failure, None on success (including "already exists")."""
    if out_path.exists():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src_path),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-acodec", "libmp3lame", "-q:a", "4", "-f", "mp3",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        tmp_path.unlink(missing_ok=True)
        return e.stderr.decode("utf-8", "replace")
    tmp_path.replace(out_path)
    return None


def verify_source_durations(audio_dir: Path, ids: set[str]) -> None:
    """Sanity-check that each source mp3's actual duration roughly matches
    the (long_end - long_start) encoded in its filename, per the naming
    convention <id>_<long_start>_<long_end>.mp3."""
    for item_id in sorted(ids):
        path = audio_dir / f"{item_id}.mp3"
        if not path.exists():
            print(f"WARNING: source audio missing: {path}", file=sys.stderr)
            continue
        _, long_start, long_end = parse_id(item_id)
        expected = long_end - long_start
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                check=True, capture_output=True, text=True,
            )
            actual = float(out.stdout.strip())
        except (subprocess.CalledProcessError, ValueError) as e:
            print(f"WARNING: could not probe duration of {path}: {e}", file=sys.stderr)
            continue
        if abs(actual - expected) > 2.0:
            print(
                f"WARNING: {path.name} duration {actual:.2f}s does not match "
                f"filename-implied duration {expected:.2f}s",
                file=sys.stderr,
            )


# ---------------------------------------------------------------------------
# Highlighting / language selection
# ---------------------------------------------------------------------------


def bold_html(text: str, span: tuple[int, int] | None) -> str:
    if span is None:
        return html.escape(text)
    start, end = span
    return (
        html.escape(text[:start])
        + "<b>" + html.escape(text[start:end]) + "</b>"
        + html.escape(text[end:])
    )


def targets_for_lang(it: dict, lang: str) -> list[dict]:
    return [t for t in it.get("targets", []) if t.get("tgt_lan") == lang]


def pick_lang_text_and_highlight(it: dict, lang: str) -> tuple[str, tuple[int, int] | None]:
    """(base_text, highlight) for one language of one item. `text`/span/
    span_start/span_end were already computed once by
    30-merge-annotations.py's build_lang_targets() against the same pooled
    "longest good+bad rendering" text stored in `automatic_texts`, so we
    just reuse them here instead of re-deriving anything."""
    base_text = it.get("automatic_texts", {}).get(lang, "")
    bad = targets_for_lang(it, lang)
    if not base_text and bad:
        base_text = max(bad, key=lambda t: len(t["text"]))["text"]
    # Only real, located matches (span_matched) count for highlighting -- a
    # target whose quote couldn't be confidently located falls back to
    # spanning the whole text (see find_span() in 30-merge-annotations.py),
    # which must NOT be treated as "highlight everything".
    matched = [t for t in bad if t.get("span_matched")]
    highlight = None
    if matched:
        best = max(matched, key=lambda t: t["span_end"] - t["span_start"])
        highlight = (best["span_start"], best["span_end"])
    return base_text, highlight


def lang_available(it: dict, lang: str) -> bool:
    return bool((it.get("automatic_texts", {}).get(lang) or "").strip()) or bool(targets_for_lang(it, lang))


def lang_bad_score(it: dict, lang: str) -> int:
    return len({t.get("annotator_model") for t in targets_for_lang(it, lang)})


def select_languages(it: dict) -> list[str]:
    en_included = bool(targets_for_lang(it, "en"))
    non_en_avail = [lang for lang in NON_EN_LANGS if lang_available(it, lang)]
    non_en_sorted = sorted(
        non_en_avail,
        key=lambda lang: (-lang_bad_score(it, lang), NON_EN_PRIORITY[lang]),
    )
    need = 2 if en_included else 3
    selected = set(non_en_sorted[:need])
    if en_included:
        selected.add("en")
    return [lang for lang in LANG_DISPLAY_ORDER if lang in selected]


# ---------------------------------------------------------------------------
# Item construction
# ---------------------------------------------------------------------------


def audio_out_path(it: dict, data_dir: Path) -> Path:
    numeric_id, long_start, _ = parse_id(it["id"])
    pad_start, pad_end = padded_span(it)
    new_start = long_start + pad_start
    new_end = long_start + pad_end
    filename = f"{numeric_id}_{new_start:.2f}_{new_end:.2f}.mp3"
    return data_dir / "assets" / filename


def clip_start_offset(it: dict) -> float:
    """Where the flagged span actually starts within the cut (padded) clip,
    i.e. how much of the CONTEXT_PAD_SECONDS leading padding survived
    clamping to the recording's own start. Used as the player's initial
    (paused) playback position, via a #t= media-fragment, so pressing play
    jumps straight to the flagged span instead of the padding before it."""
    pad_start, _ = padded_span(it)
    return max(0.0, it["orig_start"] - pad_start)


def build_item(it: dict, audio_url: str) -> dict | None:
    langs = select_languages(it)
    if not langs:
        return None
    tgt = {}
    for lang in langs:
        base_text, span = pick_lang_text_and_highlight(it, lang)
        tgt[lang] = bold_html(base_text, span)
    transcript_html = f'<b>Source (English) transcript:</b> {html.escape(it["gold_transcript"])}'
    # Media Fragments URI (#t=<seconds>): sets the player's initial playback
    # position without autoplaying -- it stays paused until the user presses
    # play, at which point it starts from that offset (native browser
    # behavior, no JS needed).
    offset = clip_start_offset(it)
    audio_url_with_offset = f"{audio_url}#t={offset:.2f}"
    src_html = f'<audio controls src="{html.escape(audio_url_with_offset)}" type="audio/mpeg"></audio>'
    return {
        "item_id": it["doc_id"],
        "instructions": transcript_html,
        "src": src_html,
        "tgt": tgt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged-errors", default=str(SCRIPT_DIR / "35-merged-annotations.jsonl"))
    parser.add_argument(
        "--audio-dir",
        default=str(SCRIPT_DIR / "input" / "no-more-embarrassing-errors-dominik" / "earnings-25" / "testset-segmented" / "audio"),
    )
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "data"), help="Pearmut root's data/ dir; clips go to <data-dir>/assets/")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "merged_errors_pearmut.json"))
    parser.add_argument("--campaign-id", default="nmee_earnings25_error_validation")
    parser.add_argument("--items-per-screen", type=int, default=10, help="Items shown together on one annotation screen (one Pearmut document)")
    parser.add_argument("--screens-per-link", type=int, default=20, help="Screens (documents) per annotator link (one Pearmut task)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N valid items (for quick testing)")
    parser.add_argument("--skip-audio", action="store_true", help="Skip cutting audio (assume clips already exist)")
    parser.add_argument("--skip-duration-check", action="store_true", help="Skip the upfront ffprobe sanity check of source files")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    data_dir = Path(args.data_dir)

    all_items = [
        json.loads(line)
        for line in Path(args.merged_errors).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {len(all_items)} merged-annotations items")

    valid_items = [it for it in all_items if is_valid_span(it)]
    skipped = len(all_items) - len(valid_items)
    print(f"{len(valid_items)} items have a usable start/end ({skipped} skipped: missing/invalid timing)")

    if args.limit is not None:
        valid_items = valid_items[: args.limit]

    if not args.skip_duration_check:
        verify_source_durations(audio_dir, {it["id"] for it in valid_items})

    # Cut audio clips (deduplicated by output path; parallelized).
    jobs = {}  # out_path -> (src_path, start, end)
    for it in valid_items:
        out_path = audio_out_path(it, data_dir)
        pad_start, pad_end = padded_span(it)
        jobs.setdefault(out_path, (audio_dir / f"{it['id']}.mp3", pad_start, pad_end))

    if not args.skip_audio:
        print(f"cutting {len(jobs)} audio clips into {data_dir / 'assets'} ...")
        failures = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(cut_audio, src_path, start, end, out_path): out_path
                for out_path, (src_path, start, end) in jobs.items()
            }
            for fut in as_completed(futures):
                out_path = futures[fut]
                err = fut.result()
                if err:
                    failures += 1
                    print(f"WARNING: failed to cut {out_path.name}: {err.strip()}", file=sys.stderr)
        print(f"audio cutting done ({failures} failures)")

    # Build Pearmut items.
    items = []
    skipped_no_lang = 0
    for it in valid_items:
        out_path = audio_out_path(it, data_dir)
        # Relative (not "/assets/...") so it resolves under a reverse-proxy
        # subpath (e.g. /pearmut/obo/) instead of the domain root, where it
        # would 404 -- an absolute path is exactly why players used to show
        # a 0:00 duration and refuse to play.
        audio_url = f"assets/{out_path.name}"
        item = build_item(it, audio_url)
        if item is None:
            skipped_no_lang += 1
            continue
        items.append(item)

    print(f"built {len(items)} Pearmut items ({skipped_no_lang} skipped: no candidate language)")

    # Group into screens (documents): items-per-screen items shown together,
    # advanced past with one Next click. Then group screens into links
    # (tasks): screens-per-link screens per annotator URL.
    screens = [
        items[i : i + args.items_per_screen]
        for i in range(0, len(items), args.items_per_screen)
    ]
    pages = [
        screens[i : i + args.screens_per_link]
        for i in range(0, len(screens), args.screens_per_link)
    ]
    print(
        f"grouped {len(items)} items into {len(screens)} screens of up to "
        f"{args.items_per_screen} items, paginated into {len(pages)} links of up to "
        f"{args.screens_per_link} screens each"
    )

    campaign = {
        "info": {
            "assignment": "task-based",
            "protocol": "ESA",
            "shuffle": False,
            "show_model_names": True,
            "instructions": PROTOCOL_INSTRUCTIONS_ESA,
            # This is error-*validation* work, not scoring: annotators are
            # confirming/rejecting spans an automatic pipeline already
            # flagged, so requiring every span/score to be set before
            # Next unlocks would force busywork on items with nothing to
            # confirm. Pearmut defaults this to true (require full
            # annotation); we opt out for this campaign only.
            "require_full_annotation": False,
        },
        "campaign_id": args.campaign_id,
        "data": pages,
    }

    Path(args.output).write_text(json.dumps(campaign, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
