#!/bin/bash

# every dataset must have this script. It will be used by processing scripts, e.g. by those in ../30_segment-align:
# source $path_to/earnings25_paths.sh 

# then, the script inside ../30_segment-align will use the following variables:

_rootpath="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."  # (except this one, it's the auxiliary variable for the next ones) 
_rootpath="../"  # (except this one, it's the auxiliary variable for the next ones) 

# directory of original audio. There are many *.mp3 files (or *.wav, or ...), all of them belong to the dataset
orig_audio_dir=$_rootpath/earnings-25/testset-segmented/audio/

# list of all audio files
# the scripts should preferably use this list. Because then we can simply have a subdataset with a shorter list, e.g. for debugging on few files
orig_audio_list=$(find $orig_audio_dir -name \*mp3)

# this is jsonl from orig earnings, it has id and transcript. 
# TODO: Other datasets have different format, we should convert them into uniform one.
gold_json=$_rootpath/earnings-25/testset-segmented/data.jsonl

# to be created by ../20_systems
# it's a directory with many *.en.jsonl files for english ASR, *.cs.jsonl for ST into target langs.
canary_outputs_dir=$_rootpath/outputs/earnings-25/
canary_asr_list=$(find $canary_outputs_dir -name \*.en.jsonl)

canary_target_langs="cs
de
en
es
fr
hu
it
pl
pt
ro
sk
uk"

# to be created by ../30_segment-align
segmented_audio_dir=$_rootpath/30_segment-align/earnings25/audio
# TODO: add if needed
#segmented_audio_list=$(find $segmented_audio_dir -name \*wav) # 


# this will include gold transcript and reference, if available for the dataset
segmented_systems_dir=$_rootpath/30_segment-align/earnings25/canary
# TODO: add if needed
# ...list
segmented_systems_merged=$_rootpath/30_segment-align/earnings25/canary-merged.jsonl
#segmented_systems_merged_dir=$_rootpath/30_segment-align/earnings25/canary-merged