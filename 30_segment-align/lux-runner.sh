#!/bin/bash -v

if [ -z "$1" ]; then
    dataset=luxparl
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

mkdir -p $segmented_systems_dir
#python3 10_asr_sentences.py $canary_outputs_dir $segmented_systems_dir --suffix ".$srclan.jsonl" --lang $mosessentlan
mkdir -p $segmented_audio_dir $segmented_audio_dir/debug
#python3 20_segment_audio.py --audio-dir $orig_audio_dir --translations-dir $segmented_systems_dir $segmented_audio_dir  --suffix ".$srclan.jsonl" --debug-prints $segmented_audio_dir/debug



if [ -f $segmented_systems_merged.ok ] ; then
    echo "File $segmented_systems_merged.ok already exists, skipping creation."
else
    : > $segmented_systems_merged
    b=$(mktemp)
    for asr in $segmented_systems_dir/*.$srclan.jsonl ; do
    #    b=$(basename $asr)
    #    b=$segmented_systems_merged_dir/${b%.en.jsonl}.jsonl
        echo "Processing $asr"
        python 30_add_and_align_sentences.py --input $asr \
            --output $b --name luxasr --lan $srclan --audio-dir $segmented_audio_dir \
            --dataset $dataset --src-language $srclan \
            --segmented-by lux-asr+moses-de+min1sec --with-words --overwrite
        cat $b >> $segmented_systems_merged
    done
    touch $segmented_systems_merged.ok
    rm $b
fi


for tgt in $canary_target_langs ; do
    echo "Processing $tgt"
    add=../20_systems/lb/*.$tgt.jsonl
    python 30_add_and_align_sentences.py --input $segmented_systems_merged \
        --output $segmented_systems_merged --name luxmt_$tgt --lan $tgt \
        --align lines --align-to luxasr --overwrite \
        --add $add
done

#if [ $dataset == "earnings-25" ]; then
#    python 35_add_earnings_gold.py --input $segmented_systems_merged \
#        --output $segmented_systems_merged --gold $gold_json \
#        --align-to luxasr
#else
#    echo "No gold data for dataset $dataset, skipping."
#fi
