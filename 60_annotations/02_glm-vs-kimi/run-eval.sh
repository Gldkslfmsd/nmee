#!/bin/bash
for x in GLM Kimi ; do
    python3 ../pearmut_to_jsonl.py  --input llm-outputs/$x""_cs.jsonl --pearmut data/annotations/$x""_cs.jsonl -o $x.jsonl
    python3 ../eval.py $x.jsonl --input-segments ../select-models/input.jsonl | tee $x.txt
done
