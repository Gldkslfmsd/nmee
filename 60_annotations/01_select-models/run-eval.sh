#!/bin/bash
for x in GLM Kimi gpt-oss-120B ; do
    python3 ../pearmut_to_jsonl.py  --input ../../40_preselect/out3/$x""_cs.jsonl --pearmut data/annotations/$x""_cs.jsonl -o $x.jsonl
    python3 ../eval.py $x.jsonl --input-segments ../../40_preselect/out3.jsonl | tee $x.txt
done
