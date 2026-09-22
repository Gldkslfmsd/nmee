#!/bin/bash

#python3 add_and_align_sentences.py \
#    --input out-asr.jsonl --output o.jsonl \
#    --document abcdocument --dataset earnings-25 --src-language en --segmented-by canary-asr+moses+gapshalved+min1sec --audio-dir audio --input-name canary_en --with-words

# create: document ID from the file name
asr=out-asr/437362344_2398.68_2995.16.en.jsonl
out=out.jsonl
python add_and_align_sentences.py --input $asr \
    --output $out --name canary_asr --lan en --audio-dir out-seg \
    --dataset earnings-25 --src-language en \
    --segmented-by canary-asr+moses+gapshalved+min1sec --with-words
# add

for tgt in de cs sk pl de ; do
    cs=../outputs/earnings-25/437362344_2398.68_2995.16.$tgt.jsonl
    python add_and_align_sentences.py --input $out \
        --output $out --name canary_$tgt --lan $tgt  --add $cs --with-words \
        --align vecalign --align-to canary_asr --overwrite \
        --split-method embed --split-window 20 --split-punct-bonus 0.03 \
        --max-size 5 
done