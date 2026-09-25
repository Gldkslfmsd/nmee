#!/bin/bash -v

if [ -z "$1" ]; then
    dataset=earnings-25
    echo "No dataset specified, using default: $dataset" >&2
else
    dataset=$1
    echo "Using dataset: $dataset" >&2
fi

source ../10_datasets/${dataset}_paths.sh

# Dominik uses the python3 venv named p3.12, others may have it different
if [ "$USER" == "machacek" ]; then
    source p3.12/bin/activate 
fi

#python3 10_asr_sentences.py $canary_outputs_dir $segmented_systems_dir
#python3 20_segment_audio.py --audio-dir $orig_audio_dir --translations-dir $segmented_systems_dir $segmented_audio_dir 

if [ -f $segmented_systems_merged.ok ]; then
    echo "File $segmented_systems_merged.ok already exists, skipping creation."
else
    : > $segmented_systems_merged
    b=$(mktemp)
    for asr in $segmented_systems_dir/*.en.jsonl ; do
    #    b=$(basename $asr)
    #    b=$segmented_systems_merged_dir/${b%.en.jsonl}.jsonl
        echo "Processing $asr"
        python 30_add_and_align_sentences.py --input $asr \
            --output $b --name canary_asr --lan en --audio-dir $segmented_audio_dir \
            --dataset $dataset --src-language en \
            --segmented-by canary-asr+moses+gapshalved+min1sec --with-words --overwrite
        cat $b >> $segmented_systems_merged
    done
    touch $segmented_systems_merged.ok
    rm $b
fi

for tgt in $canary_target_langs ; do
    echo "Processing $tgt"
    add=$canary_outputs_dir/*.$tgt.jsonl
    python 30_add_and_align_sentences.py --input $segmented_systems_merged \
        --output $segmented_systems_merged --name canary_$tgt --lan $tgt --with-words \
        --align vecalign --align-to canary_asr --overwrite \
        --split-method embed --split-window 8  \
        --add $add
done