# convert pre-selected harmful speech-translation errors into pearmut campaign



## Process 

```bash
# 0. generate annotations.jsonl with whatever automatic method (format below)

# 1. cut all segments: segments/<doc>/<doc>.<NNNN>.wav  (NNNN = 0-based line in <doc>.en.jsonl)
python segment_audio.py segments \
    --audio-dir ../iwslt26-cs-dev/audio --translations-dir ../outputs/iwslt26-cs-dev

# 2. build the campaign (only the annotated segments; one page per document)
python make_pearmut_campaign.py annotations.jsonl --segments-dir segments \
    --translations-dir ../outputs/iwslt26-cs-dev \
    --ref-yaml ../iwslt26-cs-dev/iwslt26-cs-dev.yaml \
    --ref-cs ../iwslt26-cs-dev/iwslt26-cs-dev.cs --ref-en ../iwslt26-cs-dev/iwslt26-cs-dev.en \
    --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json
   # - this script is overspecific for Dávid's IWSLT26 dev/test sets. Other datasets don't have yaml, 
   # so this script could be adapted

# 3. load campaign.json to pearmut
pearmut add -o campaign.json
```


## `annotations.jsonl` format

One JSON object per line. Each line is **one segment flagged by LLM/other method that it contains at least one harmful error**.

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
Other fields are optional.

### Target fields (`targets[]`)

Each entry is **one error in one output**. Several errors in the same output are several entries with the
same `system` and `text`. Outputs of different systems or target languages are also separate entries.

| Field | Type | Req. | Description |
|---|---|---|---|
| `tgt_lan` | str | yes | Target language, e.g. `en` |
| `system` | str | yes | System ID, `<end-to-end>` or `<asr>+<mt>`, e.g. `whisper+gemma` |
| `text` | str | yes | The system's output for the segment |
| `reference` | str/null | no | Human reference translation |
| `span` | str | no | Erroneous text, exactly `text[span_start:span_end]` |
| `span_start`, `span_end` | int | yes | Character offsets in `text` (end exclusive) |
| `intended` | str | no | What the span should have said |
| `harm_types` | list[str] | no | One or more of: `False attribution`, `Offensive`, `Embarrassing or laughable`, `Derailing or contresens`, `Safety, health or legal risk`, `Other` |
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
