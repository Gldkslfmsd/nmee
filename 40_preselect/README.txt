
AI.úfal works:

python3 40_find_harmful_errors.py --backend einfra --input first3.jsonl --output f.jsonl --show canary_asr canary_cs --annotate canary_cs --api-base https://ai.ufal.mff.cuni.cz/api/v1 --model MAC1.Qwen3.8-Flash-Next-MLX-oQ8-MTP

## Output format:

- jsonl, one line is one flagged span in at least one target

```
{
  "document": "423057182_1193.80_1790.84",
  "dataset": "earnings-25",
  "src_language": "en",
  "audio": "../30_segment-align/earnings25/audio/423057182_1193.80_1790.84/423057182_1193.80_1790.84.0000.wav",
  "beg": 0.24,
  "end": 9.36,
  "segmented_by": "canary-asr+moses+gapshalved+min1sec",
  "asr": "which is weighted primarily towards the auto sector and some of our specialty cars that we have highly engineered.",
  "asr_system": "canary_asr",
  "gold_transcript": "which is weighted primarily towards the auto sector and some of our specialty cars that we have highly engineered.",
  "targets": [  # one or more targets with a span
    {
      "tgt_lan": "cs",
      "system": "canary_cs",
      "text": "která je zaměřena především na automobilový sektor a některé naše specializované automobily, které jsme vysoce navrhli.",
      "span": "které jsme vysoce navrhli",
      "span_start": 93,
      "span_end": 118,
      "intended": "které jsme vysoce navrhl",
      "harm_types": [
        "False attribution"
      ],
      "harmfulness": 4,
      "explanation": "The verb 'navrhli' (designed) is plural, implying the company as a collective, but the correct form should be 'navrhl' (designed) in the third person singular to match 'my' (we) as the subject. This error misrepresents the speaker's role or agency, potentially making it seem like the company is acting collectively in a way that is not accurate, which could be embarrassing or misleading in a professional context.",
      "error_source": "MT"
    }, 
    # ...more systems for the same segment can be here
  ],
  "annotator": {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "backend": "vllm",
    "level": "segment",
    "annotated": [
      "canary_cs"
    ],
    "shown": [
      "canary_asr",
      "canary_cs"
    ]
  }
}
```

