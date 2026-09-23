# Segment + Align

Segment ASR to sentences, segment audio, align target sentences to ASR.

- `10_asr_sentences.py` -- takes Canary ASR with timestamps, uses Moses sentence splitter to fix sentence segmentation (so Mr. Butan stays in one sentence), adapts the segment timestamps -- the gaps between segments were inaccurate, we split them in middle and attach to adjacent segments. Also, we apply min segment length 1 second -- we merge the shorter ones.

- `20_segment_audio.py` -- takes output of `10_...` and actually segments the audio according to the timestamps
  - use option `--debug-prints` to investigate the segmentation in Audacity (the tsv file named txt as Audacity labels file)


- `30_add_and_align_sentences.py` -- takes output of `10_...`, and adds another target language. Resegments sentences by Moses and aligns them using vecalign
  - better with GPU
  - needs the same python environment as `../20_systems`, pip install `vecalign` and `mweralign`
  - `alignments.py` -- used by `30_...`

Usage:
- To be done (to complicated, it needs a runner script)
