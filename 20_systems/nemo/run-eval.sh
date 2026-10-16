#!/bin/bash

root=../../outputs/earnings-25/

ref=$root/canary_offline/st/

for lan in fr cs ; do
    for model in EuroLLM-9B Qwen3.5-27B ; do

        hyp=$root/nemo_streaming/$lan/asr-low+$model

        python eval_latency.py --ref-dir $ref --hyp-dir $hyp -l $lan -o eval/$lan-$(basename $hyp) --limit 10 --emission first-stable
    done
done