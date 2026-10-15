This is code to reproduce Nemo IWSLT 2026 Simultaneous Shared Task.

It is a cascade:

- ASR: streaming parakeet for English.
- LLM: Qwen3.5-27B -- but it uses very large GPU
    or EuroLLM-9B -- second option
- works for En-* direction. 
(for cs-en, it uses a model that they didn't publish)
- we use the top latency parameters they have for En-De, we will apply them for all en-* directions

- output format is the same as SimulStreaming (jsonl) -- it is almost the same as offline canary in this project, but "text" instead of "segment". (We should switch to text as well, TBD later.)
- processing script: work as ../run_canary.py -- multiprocessing-safe, lockdirs, etc.

qsubmit -m 20g -M 20g "p3.12/bin/python3 nemo_streaming_parakeet.py -o ../../outputs/earnings-25/nemo_streaming/asr/low --latency low ../../earnings-25/testset-segmented/audio/*mp3 --comp-unaware " --jobname par-low

qsubmit -m 20g -M 20g "p3.12/bin/python3 nemo_streaming_parakeet.py -o ../../outputs/earnings-25/nemo_streaming/asr/high --latency high ../../earnings-25/testset-segmented/audio/*mp3 --comp-unaware " --jobname par-high

