# Earnings25 English-ASR harm annotations → Pearmut campaign

Converts [`../harm_annotation_eng_asr/`](../harm_annotation_eng_asr/) (1,999 LLM-marked harmful
errors in NVIDIA Canary's English ASR of Earnings25) into a Pearmut campaign, so the annotations can
be reviewed in the browser against the audio instead of by ear against JSON.

This is the Earnings25 counterpart of [`../iwslt+claude-to-pearmut/`](../iwslt+claude-to-pearmut/),
which is overspecific to Dávid's IWSLT26 dev set. It uses the same shared `annotations.jsonl` schema
and the same protocol block (`../custom_nmee_demo.json`).

⚠️ **ASR-only.** There is no MT stage, so the "translation" being annotated is the English transcript
itself: every target has `tgt_lan: "en"`, `system: "ASR+canary"` and `error_source: "ASR"`.

## Process

```bash
# 1. shared schema: adds segment_filename, renames confidence, normalises harm_types,
#    and computes the audio bounds each clip has to cover
python normalize_annotations.py ../harm_annotation_eng_asr/_all_clips.jsonl \
    --canary-dir ../../results/canary/en -o annotations.jsonl

# 2. cut one mp3 per annotated segment: clips/<clip_id>/<clip_id>.<NNNN>.mp3
#    (1,905 clips, ~141 MB, ~2 min; needs ffmpeg)
python cut_clips.py annotations.jsonl clips \
    --audio-dir ../../earnings25_raw/earnings-25/testset-segmented/audio

# 3. build the campaign and copy its clips into the Pearmut asset dir
python make_pearmut_campaign.py annotations.jsonl --clips-dir clips \
    --confidence harmful \
    --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json

# 4. load and serve
pearmut add -o campaign.json
pearmut run
```

Step 2 needs the 3.1 GB raw dataset. If you don't have it, fetch the pre-cut clips instead —
see `../download-large-data.sh`.

## What is in the campaign

`campaign.json` as committed is the **`--confidence harmful` stratum**: the 918 errors whose
severity is `high` or `medium`, over 894 segments in 268 documents (one document per clip).

That choice leaves out the 1,081 `low`-severity errors, which the source README flags as noisy
(mostly repeated name misspellings). The 117 `high` ones have already been checked against the audio
(`../harm_annotation_eng_asr/verdicts_high_severity_human_verified.json`), so they stay in as an
agreement check on the annotator: if they disagree with those 97 confirmed / 8 rejected / 12 unsure
verdicts, that is measurable.

Other strata are one flag away:

```bash
--severity high              # 117 errors, the already-verified subset
--severity medium            # 801 errors, the unverified middle
--confidence borderline      # 1,081 low-severity errors
# (no filter)                # all 1,999
```

### Per item

- **`src`** — an `<audio>` player for the segment, plus the Earnings25 gold transcript.
  The gold transcript is a *cleaned* reference while Canary is verbatim, and it is not always right
  (Cigna, Tetra Tech and SkyWest are places where Canary is correct and the reference is not — see
  the source README). `--no-gold` hides it, which makes the judgement audio-only but slower.
- **`tgt`** — `{"ASR+canary": <the Canary transcript of the segment>}`.
- **`error_spans`** — the marked spans, pre-highlighted for the annotator to keep, move or delete.
  **No severity is set on them.** The NMEE protocol has no severity buttons, so a severity would only
  paint a colour prior onto the screen; the underlying `severity` stays available in `targets[]`,
  which Pearmut copies verbatim into the annotation logs.
  7 of the 918 errors are pure deletions (`span == ""`): nothing can be highlighted, so they appear
  in the instructions only.
- **`instructions`** — the suggested span, what was intended, the harm types, the likely source and
  the LLM's explanation.

## Annotators

`--users N` gives N task URLs. By default they are **replicated** (everyone annotates all 894 items,
which is what you want for inter-annotator agreement); `--partition` splits the 268 documents
round-robin instead, so N people each do 1/N of the work.

```bash
python make_pearmut_campaign.py annotations.jsonl --clips-dir clips --confidence harmful \
    --users 4 --partition -o campaign.json     # ~67 documents each
```

## Notes on the conversion

The source annotations already follow the shared schema; three things had to be reconciled, all in
`normalize_annotations.py`:

1. **`segment_filename` was absent.** The IWSLT script's fallback builds it from
   `Path(filename).stem`, which here is the *recording* id (`423057182`), while the audio and the
   Canary outputs are keyed by *clip* id (`423057182_1193.80_1790.84`).
2. **`confidence` vocabulary.** `confident` → `harmful`, matching the shared schema (and the
   `--confidence high` filter in the IWSLT script, which tests for `"harmful"`).
3. **`harm_types` labels.** The short forms (`Derailing/contresens`) are mapped to the protocol enum
   (`Derailing or contresens`).

One data quirk is worth knowing about. For **32 of the 1,905 records** the stored `asr` is a
character-truncated window that runs past the end of its own Canary segment into the next one(s) —
in 16 of them the marked span sits in that overflow. Cutting on the segment boundary would have shown
the annotator text whose audio was missing. `normalize_annotations.py --canary-dir` detects this and
emits `cut_start_in_clip` / `cut_end_in_clip`, extending the clip to the end of the last segment the
text touches; `seg_*_in_clip` is left untouched. Clips also get 0.3 s of padding on each side
(`--pad`).

All 1,999 spans satisfy `text[span_start:span_end] == span` exactly, so the pre-filled highlights are
byte-correct; `normalize_annotations.py` re-checks this and fails loudly if it ever stops holding.

## Verified

`campaign.json` was loaded with `pearmut add` and served with `pearmut run` (pearmut from PyPI):
the campaign registers, the annotation page returns 200 and the clips are served from
`/assets/earnings25_en_asr_harm_review/...` as `audio/mpeg`.
