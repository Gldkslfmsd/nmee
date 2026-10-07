#!/bin/bash
for x in GLM Kimi gpt-oss-120B ; do
    python3 ../pearmut_to_jsonl.py --eval  --input ../../40_preselect/out3.jsonl --pearmut data/annotations/$x""_cs.jsonl -o $x.jsonl | tee $x.txt
done
