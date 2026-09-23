#!/bin/bash

#python3 add_and_align_sentences.py \
#    --input out-asr.jsonl --output o.jsonl \
#    --document abcdocument --dataset earnings-25 --src-language en --segmented-by canary-asr+moses+gapshalved+min1sec --audio-dir audio --input-name canary_en --with-words

# create: document ID from the file name
out=out.jsonl
#: > $out
#for asr in out-asr/*.en.jsonl ; do
#    echo "Processing $asr"
#    python add_and_align_sentences.py --input $asr \
#        --output o.jsonl --name canary_asr --lan en --audio-dir out-seg \
#        --dataset earnings-25 --src-language en \
#        --segmented-by canary-asr+moses+gapshalved+min1sec --with-words --overwrite
#    cat o.jsonl >> $out
#done

for tgt in cs sk pl de ; do
    cs=../outputs/earnings-25/*.$tgt.jsonl
    python add_and_align_sentences.py --input $out \
        --output $out --name canary_$tgt --lan $tgt --with-words \
        --align vecalign --align-to canary_asr --overwrite \
        --split-method embed --split-window 8  \
        --add $cs #--embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
done