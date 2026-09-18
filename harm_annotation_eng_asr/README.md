# Earnings25 — harmful ASR error annotations (Canary English)

1,999 harmful-error annotations over 288 of the 290 `testset-segmented` clips of **Earnings25**,
marked on the **English ASR output of NVIDIA Canary** (the pre-computed workshop outputs).

Annotated by **Augustin Barruol** (École polytechnique) — MT Marathon 2026.

⚠️ **This is ASR-only.** There is no MT stage here, so every annotation has
`"tgt_lan": "en"` and `"error_source": "ASR"`. The records use the shared workshop schema so they
can sit next to translation annotations, but the "translation" is the English transcript itself.

## Files

| File | Contents |
|---|---|
| `_all_clips.jsonl` | **all annotations** — 1,905 lines covering every clip |
| `_index.json` | all 290 clips → company, file name, number of segments/errors |
| `<clip_id>.jsonl` | the same data split per clip — 290 files, one per audio |

`_all_clips.jsonl` is exactly the concatenation of the per-clip files, so the two are interchangeable.
**The git repository carries only `_all_clips.jsonl` and `_index.json`**, to keep it to a single 3 MB
artefact instead of 290 small ones; the full export has the per-clip files as well.

### `_all_clips.jsonl`

**One line per (clip, Canary segment) that contains at least one error**, ordered by `clip_id` then
`segment`. All errors inside one segment share that line and appear in its `targets` list — so
1,905 lines carry 1,999 annotated errors.

Every line names its own clip (`clip_id`, `doc_id`), so the file is self-describing and splitting it
back per clip needs no extra metadata:

```python
import json, collections
by_clip = collections.defaultdict(list)
for line in open("_all_clips.jsonl"):
    r = json.loads(line)
    by_clip[r["clip_id"]].append(r)
```

Two of the 290 clips contribute **no lines at all**, because no harmful error was found in them
(in the per-clip export they are 0-byte files):

- `427822467_2386.28_2689.65` — Krispy Kreme, Inc.
- `443256014_2411.64_3005.64` — Exxon Mobil Corporation

Their ASR/reference differences were only spelling and filler words; in a couple of Exxon spots Canary
is arguably more correct than the reference. Both appear in `_index.json` with `"errors": 0`, so
"reviewed, nothing found" stays distinguishable from "not processed".

## Identifiers and audio

`clip_id` has the form `<recording_id>_<start>_<end>`, e.g. `432246475_2425.88_3025.34`.

- `filename` = `<recording_id>.mp3`, the full recording (`earnings-25/testset-full/`).
- `clip_filename` = `audio/<clip_id>.mp3`, the ~10-minute clip (`earnings-25/testset-segmented/`), which is what was transcribed.
- `clip_orig_start` / `clip_orig_end` = position of the clip inside the full recording.
- `orig_start` / `orig_end` = segment timing **in the full recording** (clip offset already added).
- `seg_start_in_clip` / `seg_end_in_clip` = same timing relative to the clip file.
- `segment`, `concat_segment` = 0-based index of the Canary segment; `en_file_line` = its 1-based line
  in `results/canary/en/<clip_id>.en.jsonl`.

## Text fields

- `asr` — Canary output for the segment.
- `gold_transcript` — Earnings25 reference for the same segment.

Both are rebuilt by joining word tokens with single spaces, so the character offsets below are exact
against these strings. The reference is a cleaned transcript (disfluencies removed) while Canary is
verbatim, so fillers such as "you know" appear in `asr` only and are **not** annotated as errors.

## Inside `targets[]`

- `span`, `span_start`, `span_end` — the wrong words in `asr`; `asr[span_start:span_end] == span` exactly.
  For a pure deletion `span` is `""` and both offsets mark the point where the words are missing.
- `intended`, `intended_start`, `intended_end` — the reference counterpart in `gold_transcript`.
  Empty for a pure insertion. `intended_segment` gives the segment the reference words belong to; when
  it differs from `segment` the offsets are `null` (the words fall in a neighbouring segment).
- `span_orig_start` / `span_orig_end` — timing of the wrong words in the full recording, for listening.
- `harm_types` — protocol labels; `harm_types_code` — the same as enum codes
  (`FALSE_ATTRIBUTION`, `OFFENSIVE`, `EMBARRASSING`, `DERAILING`, `SAFETY`, `OTHER`).
- `severity` — `high` / `medium` / `low`. `confidence` and `borderline` are **derived from it**
  (`low` → `borderline: true`), not an independent confidence estimate.
- `explanation` — what was meant, what the transcript says, why it is harmful.

## Counts

| Harm type | Errors |
|---|---|
| False attribution | 1,237 |
| Embarrassing/laughable | 512 |
| Derailing/contresens | 430 |
| Other | 77 |
| Safety/health/legal risk | 54 |
| Offensive | 11 |

(Sums exceed 1,999 because an error can carry several types.) Severity: 117 high, 801 medium, 1,081 low.

## How it was produced, and its limits

Reference and ASR were aligned with `jiwer` (case-folded only — punctuation, casing, `$` and decimal
points are preserved, which matters for figures such as `$6.2` vs `62`). Each clip was then read
segment by segment by an LLM (Claude Opus 5) against the workshop harm protocol; the quoted words were
located back in the transcripts automatically.

- **No one listened to the audio.** Judgements come from the text alone. Use `span_orig_start` to check.
- **The reference is not always right.** In some places Canary is correct and the reference is wrong;
  the reviewers skipped those where they noticed, but not exhaustively.
- **`low` severity is noisy** — largely minor name misspellings, often the same name repeated.
  For a high-precision subset, filter `severity == "high"`.
- Two clips have no annotations (Krispy Kreme, Exxon Mobil): their differences were only spelling or fillers.
