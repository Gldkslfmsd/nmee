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
#    (1,905 clips, ~141 MB, ~2 min; needs ffmpeg). Re-running is cheap: clips/.bounds.json
#    records the cut used for each clip, so only clips whose bounds changed are redone.
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

For the **merged** campaigns that also carry Ondřej's DSPy divergence findings and the de/cs/pl/sk
outputs, run `./build_merged_campaigns.sh` — see [Merging with the DSPy divergence set](#merging-with-the-dspy-divergence-set).

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

---

# Merging with the DSPy divergence set

`merge_annotations.py` combines this set with the one from
[`../dspy-search-for-errors/`](../dspy-search-for-errors/) (Ondřej Bojar's DSPy divergence finder:
three LLMs — GLM, Kimi, gpt-oss-120B — voting over the same 290 clips in five languages).

```bash
./build_merged_campaigns.sh          # downloads his two archives, runs his merge, then ours
```

The whole thing is reproducible from what is committed: the two archives are fetched from the URLs
in `../dspy-search-for-errors/*.url`, and step 1 is *his* `30-merge-annotations.py`, not a
reimplementation — it already emits the shared `annotations.jsonl` schema.

## What had to be reconciled

The two sets describe the same audio but agree on almost nothing else:

| | `harm_annotation_eng_asr/` | `dspy-search-for-errors/` |
|---|---|---|
| unit | one Canary **en** segment | a gold-transcript word range, mapped to time |
| languages | en only | en + de/cs/pl/sk |
| offsets | exact, into `asr` | into the model's own quoted rendering |
| labels | 6 harm types + high/medium/low | `harm_types` always `["Other"]`; error class inside the explanation |
| annotator | one LLM pass | three LLMs voting → `num_models_reporting` |

`merge_annotations.py` puts both on the **Canary English segment grid**, which is the unit this
directory already uses and the unit the audio clips are cut to. Each divergence record is attached
to the Canary segment it overlaps most (91% of them touch exactly one).

- **English is a real merge.** A divergence record's `automatic_texts["en"]` is a window of the same
  Canary transcript, so its spans are re-located inside the full segment text and land on the *same*
  target as this set's spans. That is where the two sets can actually be compared: **1,210 segments
  are flagged by both.**
- **de/cs/pl/sk are carried over, not merged.** There is nothing here to merge them with, and they
  cannot be put on the English grid either: the per-language Canary outputs are independently
  segmented (88/84/89/90/87 segments for the same clip), so an English segment number does not
  index them. The divergence record's own time-aligned `automatic_texts[lang]` is used as the
  target text, taking the largest-overlapping window when several map to one segment and
  re-locating the other windows' spans into it.

Nothing is dropped. A span that cannot be re-located keeps its explanation and simply is not
pre-highlighted; the three models are collapsed per (language, span) with `num_models_reporting`
kept on the target.

## The merged set

At `--min-models 2` (an error at least two of the three LLMs agreed on):

```
asr-harm targets in:      1999
divergence records kept:  2511
divergence spans:         13185 -> 4362 re-located, 168 not found (kept, not pre-highlighted)
3104 segments, 15184 targets, 1210 segments flagged by both sets
```

Of the 3,104 segments: 695 come from this set only, 1,199 from the divergence set only, 1,210 from
both. The remaining 8,655 divergence spans flag a whole output rather than a span — that is how the
source data is, and they appear in the instructions without a highlight.

`--min-models 1` keeps everything (5,223 records) and `--min-models 3` only unanimous findings (699).

## The campaigns

One campaign per **annotator language pair** — English plus one target language — because nobody
reads en+de+cs+pl+sk, and every one of these is annotatable by an en+X reader:

| campaign | documents | items | errors | pre-filled spans |
|---|---|---|---|---|
| `earnings25_merged_en` | 289 | 2,206 | 3,185 | 2,104 |
| `earnings25_merged_ende` | 289 | 2,410 | 4,110 | 2,161 |
| `earnings25_merged_encs` | 290 | 2,460 | 4,395 | 2,198 |
| `earnings25_merged_enpl` | 290 | 2,476 | 4,397 | 2,190 |
| `earnings25_merged_ensk` | 289 | 2,373 | 4,025 | 2,159 |

Only `campaign_en.json` is committed (3.3 MB) — it is the English both-sources merge, the part that
is new rather than a carry-over. The other four are one `./build_merged_campaigns.sh` away and are
gitignored, as is `merged_annotations.jsonl` (16 MB).

Three campaign settings differ from the ASR-only one, because the output columns are now languages
rather than competing systems:

- `--shuffle off` — Pearmut shuffles model columns per item by default, to avoid positional bias.
  With languages as columns that only disorients the annotator.
- `--show-model-names on` — otherwise the columns are unlabelled, and an annotator cannot tell
  which language they are reading.
- `--slim` — drops the copy of `targets`/`asr`/`gold_transcript` that Pearmut would otherwise keep
  in the logs. It is useful provenance but roughly quadruples the file; `item_id` joins back to
  `merged_annotations.jsonl` anyway.

Filters compose, so any other slice is one flag away:

```bash
--lang cs --source dspy-divergencies   # Czech, divergence findings only
--lang en --source asr-harm            # the original ASR-only campaign, from the merged file
--min-models 3                         # only what all three LLMs agreed on
--severity high                        # applies to asr-harm targets; others pass through
```

## A caveat about `clips/`

The clips are cut to cover everything an item displays, so a segment that also carries a divergence
window gets a wider clip than the ASR-only campaign would need. Bounds only ever widen, so building
the merged set last (which `build_merged_campaigns.sh` does) leaves every campaign with audio that
covers at least what it shows. `clips/.bounds.json` records what each clip was cut to.

## Verified

`campaign_encs.json` (en+cs, 2,460 items) was loaded with `pearmut add` and served: the annotation
page returns 200, clips are served as `audio/mpeg`, and a merged item shows the English ASR and the
Czech output side by side with pre-filled spans on both and instructions labelled per system.
