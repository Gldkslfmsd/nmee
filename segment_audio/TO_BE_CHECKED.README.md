# Harmful speech-translation errors → Pearmut

Tools for reviewing suggested harmful errors in long-form speech translation (dataset `iwslt26-cs-dev`,
Czech speech → English) in [Pearmut](https://github.com/zouharvi/pearmut).

| File | Purpose |
|---|---|
| `annotations.jsonl` | Suggested harmful errors, one line per segment (format below) |
| `asr_sentences.py` | Re-segments Canary output into sentences (Moses sentence splitter) |
| `add_and_align_sentences.py` | Builds the sentence-level, multi-system JSONL (`--align mwer/vecalign/time`) |
| `setup_aligners.sh` | Adds and builds the aligners as git submodules under `third_party/` |
| `segment_audio.py` | Cuts every long audio file into per-segment clips |
| `make_pearmut_campaign.py` | Builds a Pearmut campaign from `annotations.jsonl` |

## Aligners (git submodules)

`add_and_align_sentences.py` uses two external aligners, kept as git submodules so their versions are
pinned and the alignments stay reproducible:

    third_party/vecalign     https://github.com/thompsonb/vecalign    --align vecalign
    third_party/mweralign    https://github.com/mjpost/mweralign      --align mwer

    ./setup_aligners.sh                     # add/clone and build both
    git submodule update --init --recursive # on another machine, then run setup_aligners.sh again

Vecalign is not on PyPI and needs its Cython extension built in place, which the script does. Both are
imported from `third_party/` if present, otherwise from an installed package of the same name;
`--vecalign-dir` / `--mweralign-dir` point elsewhere.

## Pipeline

```bash
# 1. cut all segments: segments/<doc>/<doc>.<NNNN>.wav  (NNNN = 0-based line in <doc>.en.jsonl)
python segment_audio.py segments \
    --audio-dir ../iwslt26-cs-dev/audio --translations-dir ../outputs/iwslt26-cs-dev

# 2. build the campaign (only the annotated segments; one page per document)
PYTHONUTF8=1 python make_pearmut_campaign.py annotations.jsonl --segments-dir segments \
    --translations-dir ../outputs/iwslt26-cs-dev \
    --ref-yaml ../iwslt26-cs-dev/iwslt26-cs-dev.yaml \
    --ref-cs ../iwslt26-cs-dev/iwslt26-cs-dev.cs --ref-en ../iwslt26-cs-dev/iwslt26-cs-dev.en \
    --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json

# 3. load it
pearmut add -o campaign.json
```

### `segment_audio.py`

Reads `start`/`end` (seconds) from each `<doc>.en.jsonl` and writes one WAV per segment. It uses Python's
`wave` module and falls back to `sox` if a file can't be read that way. Option: `--pad SECONDS`.

### `make_pearmut_campaign.py`

- **Protocol:** the `info` block of `../custom_nmee_demo.json` (path relative to the working directory),
  copied verbatim. That is ESA with no sliders or severities, word-level spans, and the harm-annotation
  instructions. `--template FILE` takes `info` from another campaign file.
- **Item:** the source is the audio clip, the gold transcript and the reference translation. The target is
  one text per system. The item's `instructions` show each suggestion: span, intended meaning, harm types,
  confidence, likely error source and explanation.
- **Pre-filled spans:** off by default; `--prefill` pre-highlights suggested spans with severity `major` (red).
  `--prefill-severity` changes the value.
- **Clip URLs:** `./assets/<campaign-id>/<segment_filename>`. `--copy-assets` copies the clips to
  `$PEARMUT_ROOT/data/assets/<campaign-id>/`.
- **Gold and reference:** by default, the values stored in `annotations.jsonl` are shown. With
  `--translations-dir` plus `--ref-yaml/--ref-cs/--ref-en`, they are re-aligned by time overlap with the
  reference segmentation instead. A reference segment is used if it overlaps the clip by more than
  `--min-overlap` seconds (default 0.1) and covers at least `--min-overlap-ratio` (default 0.3) of the
  clip or of itself. Reference segments are whole sentences, so they can still cover more than the clip.
- **Filtering and display:**
  - `--confidence high|low|all` keeps only harmful, only borderline, or all targets.
  - `--no-gold` and `--no-reference` hide those texts.
  - `--show-asr` also shows the ASR text.
- **Annotators and IDs:**
  - `--users N` makes N identical tasks, i.e. N annotator links (default 4).
  - `--campaign-id` sets the campaign ID. Use a different one for each campaign so asset folders don't collide.
- **Extra fields:** all annotation fields are stored as extra item keys, so they end up in Pearmut's logs.

## `annotations.jsonl` format

One JSON object per line. Each line is **one segment containing at least one harmful error**.

### Segment fields

| Field | Type | Req. | Description |
|---|---|---|---|
| `doc_id` | str | yes | `<dataset>/<audio stem>`, e.g. `iwslt26-cs-dev/5332.s063067.r4` |
| `filename` | str | yes | Long audio file, e.g. `5332.s063067.r4.wav` |
| `segment_filename` | str | * | Clip path relative to the segments directory, e.g. `./5332.s063067.r4/5332.s063067.r4.0006.wav` |
| `segment` | int | * | 0-based order of the segment in the document (line in `<doc>.en.jsonl`) |
| `orig_start`, `orig_end` | float | * | Segment time span in the long audio (seconds) |
| `asr` | str | ** | ASR transcript of the segment |
| `gold_transcript` | str | ** | Gold (human) transcript |
| `targets` | list | yes | Annotated outputs, see below |

\* At least one way of locating the segment: `segment_filename`, `segment`, or `orig_start` + `orig_end`.
\*\* At least one of `asr` and `gold_transcript`.

### Target fields (`targets[]`)

Each entry is **one error in one output**. Several errors in the same output are several entries with the
same `system` and `text`. Outputs of different systems or target languages are also separate entries.

| Field | Type | Req. | Description |
|---|---|---|---|
| `tgt_lan` | str | yes | Target language, e.g. `en` |
| `system` | str | yes | System ID, `<asr>+<mt>`, e.g. `ASR+canary-v2-1B` |
| `text` | str | yes | The system's output for the segment |
| `reference` | str/null | no | Human reference translation |
| `span` | str | yes | Erroneous text, exactly `text[span_start:span_end]` |
| `span_start`, `span_end` | int | yes | Character offsets in `text` (end exclusive) |
| `intended` | str | yes | What the span should have said |
| `harm_types` | list[str] | yes | One or more of: `False attribution`, `Offensive`, `Embarrassing or laughable`, `Derailing or contresens`, `Safety, health or legal risk`, `Other` |
| `confidence` | str | no | Confidence of the (LLM) marker: `harmful` (high) or `borderline` (low) |
| `borderline` | bool | no | `true` iff `confidence == "borderline"` |
| `explanation` | str | no | Why the error is harmful |
| `error_source` | str | no | `ASR`, `MT`, `ASR+MT` or `unknown` |
| `en_file_line`, `concat_segment` | int | no | Provenance: line in `csen_dev_en.txt` (with headers) / 1-based segment index without headers |

### Example

```json
{"doc_id": "iwslt26-cs-dev/5332.s063067.r4", "filename": "5332.s063067.r4.wav",
 "segment_filename": "./5332.s063067.r4/5332.s063067.r4.0006.wav", "segment": 6,
 "asr": "že tato novela zákona o provozu na pozemních komunikacích jde vlastně paralelně s jinou novelou, …",
 "gold_transcript": "Teď tedy v roli zpravodaje pouze konstatuji, že tato novela zákona o provozu …",
 "targets": [{"tgt_lan": "en", "system": "ASR+canary-v2-1B",
   "text": "this amendment to the ground handling law is actually parallel to another amendment, it is law No.",
   "reference": "At this moment, as rapporteur, I only note that this amendment to the Road Traffic Act …",
   "span": "ground handling law", "span_start": 22, "span_end": 41,
   "intended": "Road Traffic Act", "harm_types": ["False attribution"],
   "confidence": "borderline", "borderline": true,
   "explanation": "Names the wrong law (an aviation term); the rest of the speech makes clear it is about the drivers' points system.",
   "error_source": "MT", "en_file_line": 10, "concat_segment": 7}]}
```

(The example is pretty-printed; in the file each record is on one line.)

## Current annotations

- **Size:** 64 segments with 73 targets (37 harmful, 36 borderline) in 29 documents.
- **Source:** suggested by Claude from the ASR output (`csen_dev_cs.txt`), the translation (`csen_dev_en.txt`),
  the gold transcript and the reference, following the harmful-error annotation manual. They are
  suggestions for human review, not gold labels.
- **Alignment:** timestamps were not available, so `asr`, `gold_transcript` and `reference` were aligned by
  text similarity. `asr` is the ASR line containing the error, and the ASR segmentation differs from the MT
  segmentation. `reference` is `null` where no reference line matched reliably (21 segments). Re-align by
  time with `--ref-yaml` when possible.

## Known issues

- **Reference shift:** `iwslt26-cs-dev.en` is locally shifted against `iwslt26-cs-dev.cs`. From line 33
  (document 5899), one English line covers two Czech lines; document 6726 shows a similar shift.
- **Document `6807.s045064.r2`:** the ASR and the translation contain a different speech than the gold
  transcript. It is not annotated.
- **Encoding:** on machines with a non-UTF-8 locale, run with `PYTHONUTF8=1` or open files with
  `encoding="utf-8"`.
