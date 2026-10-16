#!/bin/bash
base=../../outputs/earnings-25/nemo_streaming
for lat in low high ; do
    for lang in fr cs; do
      out="$base/$lang/asr-$lat+EuroLLM-9B"
      mkdir -p "$out"
      p3.12/bin/python3 nemo_streaming_llm.py -t $lang \
          --nmt-model utter-project/EuroLLM-9B-Instruct  \
          -o "$out" $base/asr/$lat/*.en.jsonl
    done
done
