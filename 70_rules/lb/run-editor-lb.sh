#!/bin/bash
export LLM_API_KEY=$(cat ../ai.ufal_key.txt)
port=9051
for tgt in en fr de ; do
    python3 ../rule_editor.py -o lb-rules.$tgt.jsonl ../../60_annotations/20_lb/confirmed.luxparl-Kimi_en,de,fr-14.10.$tgt.jsonl --backend api --port $port --api-base https://ai.ufal.mff.cuni.cz/api/v1 --model MAC1.Qwen3.8-Flash-Next-MLX-oQ8-MTP --systems ../../30_segment-align/lb-systems/luxparl-systems.jsonl --batch-size 1 --enable-thinking low --audio-root .. &
    port=$(($port+1))
done
wait
