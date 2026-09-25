# Segment + Align

Segment ASR to sentences, segment audio, align target sentences to ASR.

- `10_asr_sentences.py` -- takes Canary ASR with timestamps, uses Moses sentence splitter to fix sentence segmentation (so Mr. Butan stays in one sentence), adapts the segment timestamps -- the gaps between segments were inaccurate, we split them in middle and attach to adjacent segments. Also, we apply min segment length 1 second -- we merge the shorter ones.

- `20_segment_audio.py` -- takes output of `10_...` and actually segments the audio according to the timestamps
  - use option `--debug-prints` to investigate the segmentation in Audacity (the tsv file named txt as Audacity labels file)


- `30_add_and_align_sentences.py` -- takes output of `10_...`, and adds another target language. Resegments sentences by Moses and aligns them using vecalign
  - better with GPU
  - needs the same python environment as `../20_systems`
  - install vecalign: `git clone https://github.com/thompsonb/vecalign ; cd vecalign; uv pip install . -e `
  - install mweralign: `uv pip install mweralign`
  - `alignments.py` -- used by `30_...`


## Format

- jsonl, one line is one sentence

```
{
  "document": "423057182_1193.80_1790.84",
  "dataset": "earnings-25",
  "src_language": "en",
  "systems_info": {
    "canary_asr": {
      "lan": "en",
      "is_human": false
    },
    "canary_cs": {
      "lan": "cs",
      "is_human": false
    },
    ...
    "gold_transcript": {
      "lan": "en",
      "is_human": true
    }
  },
  "segmented_by": "canary-asr+moses+gapshalved+min1sec",  # how it was segmented and to what
  "audio": "../30_segment-align/earnings25/audio/423057182_1193.80_1790.84/423057182_1193.80_1790.84.0000.wav",
  "beg": 0.24,
  "end": 9.36,
  "text": {
    "canary_asr": "which is weighted primarily towards the auto sector and some of our specialty cars that we have highly engineered.",
    "canary_cs": "která je zaměřena především na automobilový sektor a některé naše specializované automobily, které jsme vysoce navrhli.",
    ...
    "gold_transcript": "which is weighted primarily towards the auto sector and some of our specialty cars that we have highly engineered."
  },
  "words": {   # not mandatory
    "canary_asr": [
      {
        "word": "which",
        "start_offset": 3,
        "end_offset": 4,
        "start": 0.24,
        "end": 0.32
      },
        ...
    ]
    ...
  },
  "alignment": {  # how it was aligned to canary_asr
    "canary_cs": "vecalign 1:1",
    "gold_transcript": "mwer"
  },
  "gold_meta": {  # not mandatory. It's meta info from earnings gold
    "participants": [
      "Operator",
      "Justin M. Roberts",
      "Lorie L. Tekorius",
      "Brian J. Comstock",
      "Michael Donfris",
      "Ken Hoexter",
      "Bascome Majors"
    ],
    "extra_fields": {
      "Company": "The Greenbrier Companies, Inc.",
      "Country": "US",
      "ReleaseDate": "2025-01-08T22:00:00",
      "Industry": "Railroad Rolling Stock",
      "MarketCap": "1525158728.94"
    }
  }
}
```
