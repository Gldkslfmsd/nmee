#!/bin/bash

# Dominik uses the python3 venv named p3.12, others may have it different
if [ "$USER" == "machacek" ]; then
    source ../p3.12/bin/activate 
fi

out=luxparl-systems.jsonl
python3 ../40_skim_jsonl.py < ../luxparl-segmented/luxparl-merged.jsonl > luxparl-systems.jsonl

d=../../20_systems/lb/with-context
for tgt in de fr en ; do
    python ../30_add_and_align_sentences.py --input $out \
        --output $out --name luxmt-with-context_$tgt --lan $tgt \
        --align mwer --align-to luxmt_$tgt  \
        --add $d/*$tgt.jsonl
done
