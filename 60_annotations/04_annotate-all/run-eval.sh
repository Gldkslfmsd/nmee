#!/bin/bash
x=dummy
python3 ../pearmut_to_jsonl.py  --input head-100-dummy.jsonl --pearmut data/annotations/$x""_cs.jsonl -o $x.jsonl
python3 ../eval.py $x.jsonl --input-segments dummy-shuf.jsonl | tee $x.txt
python3 ../eval-recall.py $x.jsonl --input-segments dummy-shuf.jsonl | tee $x-recall.txt
