#!/bin/bash
source p3.12/bin/activate
base=../../outputs/earnings-25/nemo_streaming
for lat in high low ; do
    for lang in pl cs fr; do
      out="$base/$lang/asr-$lat+Qwen3.5-27B"
      mkdir -p "$out"
      python3 nemo_streaming_llm.py -t $lang \
          --nmt-model Qwen/Qwen3.5-27B --max-model-len 131072 --llm-param 'limit_mm_per_prompt={"image":0,"video":0}' \
          -o "$out" $base/asr/$lat/*.en.jsonl
    done
done
