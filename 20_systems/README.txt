Canary-v2:
==========
==========

run_canary.py
- it's self-documented
- Dominik installed it with python3.12 venv -- older didn't work
- then: 
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 ; uv pip install -U nemo_toolkit['asr']
    - cu128 is requirement for úfal cluster, use whatever works elsewhere

Usage (example):
================

p3.12/bin/python3 run_canary.py earnings-25/testset-segmented/audio/*mp3 -s en -t de es ro pt hu uk -o outputs/earnings-25

- it writes outputs to -o dir

- input->output filenames:

 INPUT:  442982074_598.92_1199.00.mp3 
 OUTPUT: 442982074_598.92_1199.00.pt.jsonl  --> .pt. is the target language code, in this case Portuguese (the Canary-v2 one, which is European)
         442982074_598.92_1199.00.en.jsonl  --> .en. is the same language as the source (-s en), so canary-v2 is used as en ASR

Output format:
==============

- it is jsonl
- one line is one output segment from Canary. (It will be post-edited in the following step -- because the sentence segmentation and gaps between segments are inaccurate, and we need some min segment length).
- it looks like this:

{
  "segment": "Felhívjuk figyelmét arra, hogy ezek az előretekintő kijelentések a tizenkilencvenötös magántőke-peres peres reformtörvény biztonsági kikötő rendelkezései alá tartoznak, és az NTIC szeretné kihasználni a biztonsági kikötő védelmét ezekre a kijelentésekre.",
  "start_offset": 600,  # not mandatory for other systems
  "end_offset": 761,    # not mandatory for other systems
  "start": 48.0,   # in seconds from begining of audio
  "end": 60.88,    # in seconds from begining of audio
  "words": [  # not mandatory if other system does not have them
    {
      "word": "Felhívjuk",
      "start_offset": 600, # not mandatory for ther systems
      "end_offset": 606, # not mandatory for ther systems
      "start": 48.0, # in seconds form the beginning of audio
      "end": 48.480000000000004 # in seconds form the beginning of audio
    },
    ...


Other models/systems
====================
- to be done
- preferably, refactor run_canary.py, move common parts to modules (audio loading, cmdline options, loop over input/output...), create new entry point for another system family (e.g. whisper)
- or, create a new code that runs another systems separately, without re-using run_canary.py code, save outputs in whatever format, and then use a script to convert them to our jsonl format. Becuase our follow up scripts will use this format.
