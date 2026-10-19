#!/bin/bash

# every dataset must have this script. It will be used by processing scripts, e.g. by those in ../30_segment-align:
# source $path_to/luxparl_paths.sh 

# then, the script inside ../30_segment-align will use the following variables:

#_rootpath="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."  # (except this one, it's the auxiliary variable for the next ones) 
_rootpath="../"  # (except this one, it's the auxiliary variable for the next ones) 

_luxpath=$_rootpath/10_datasets/lb/

srclan=lb

# moses sentence splitter for de works the best for lb
mosessentlan=de

# directory of original audio. There are many *.mp3 files (or *.wav, or ...), all of them belong to the dataset
orig_audio_dir=$_luxpath/nmee_lb/audio

# list of all audio files
# the scripts should preferably use this list. Because then we can simply have a subdataset with a shorter list, e.g. for debugging on few files
orig_audio_list=$(find $orig_audio_dir -name \*mp3)

# this is jsonl from orig earnings, it has id and transcript. 
# TODO: Other datasets have different format, we should convert them into uniform one.
gold_json=$_luxpath/gold.jsonl

# to be created by ../20_systems
# it's a directory with many *.en.jsonl files for english ASR, *.cs.jsonl for ST into target langs.
canary_outputs_dir=$_luxpath/outputs/
canary_asr_list=$(find $canary_outputs_dir -name \*.$srclan.jsonl)

canary_target_langs="de
en
fr"

segmented_systems_dir=$_rootpath/30_segment-align/luxparl-segmented/
segmented_audio_dir=$segmented_systems_dir/audio
# TODO: add if needed
# ...list
segmented_systems_merged=$_rootpath/30_segment-align/luxparl-segmented/luxparl-merged.jsonl
#segmented_systems_merged_dir=$_rootpath/30_segment-align/earnings25/canary-merged
