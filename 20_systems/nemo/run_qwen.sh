#!/bin/bash
base=../../outputs/earnings-25/nemo_streaming
for lat in high low ; do
    for lang in cs fr; do
      out="$base/$lang/asr-$lat+Qwen3.5-27B"
      mkdir -p "$out"
      p3.12/bin/python3 nemo_streaming_llm.py -t $lang \
          --nmt-model Qwen/Qwen3.5-27B --comp-unaware \
          -o "$out" $base/asr/$lat/*.en.jsonl
    done
done
