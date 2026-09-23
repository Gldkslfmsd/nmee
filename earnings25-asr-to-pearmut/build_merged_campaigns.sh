#!/bin/bash
# Rebuild everything on the merged (ASR-harm + DSPy-divergencies) annotation set.
#
# Inputs that are not in this repo:
#   - ../dspy-search-for-errors/25-divergencies-by-llms.zip and 15-roughaligned.zip
#     (URLs in the .url files next to them)
#   - ../../earnings25_raw/earnings-25/testset-segmented/audio  (3.1 GB), or the pre-cut
#     clips from ../download-large-data.sh
set -euo pipefail
cd "$(dirname "$0")"

DSPY=../dspy-search-for-errors
WORK=${WORK:-./work}
AUDIO=${AUDIO:-../../earnings25_raw/earnings-25/testset-segmented/audio}
CANARY=${CANARY:-../../results/canary/en}
MIN_MODELS=${MIN_MODELS:-2}
mkdir -p "$WORK"

# 0. Ondrej's two archives -> $WORK/{15-roughaligned,25-divergencies-by-llms}
for name in 15-roughaligned 25-divergencies-by-llms; do
    if [ ! -d "$WORK/$name" ]; then
        wget -O "$WORK/$name.zip" "$(cat "$DSPY/$name.url")"
        unzip -q "$WORK/$name.zip" -d "$WORK"
    fi
done

# 1. his own merge step: per-model findings -> the shared annotations.jsonl schema
python "$DSPY/30-merge-annotations.py" \
    --roughaligned-dir "$WORK/15-roughaligned" \
    --divergencies-dir "$WORK/25-divergencies-by-llms" \
    --output "$WORK/35-merged-annotations.jsonl" \
    --csv-output "$WORK/35-merged-annotations.csv"

# 2. put both sets on the Canary English segment grid
python merge_annotations.py \
    --asr-harm annotations.jsonl \
    --divergencies "$WORK/35-merged-annotations.jsonl" \
    --canary-dir "$CANARY" \
    --index ../harm_annotation_eng_asr/_index.json \
    --min-models "$MIN_MODELS" \
    -o merged_annotations.jsonl

# 3. clips for every segment in the merged set (skips unchanged ones)
python cut_clips.py merged_annotations.jsonl clips --audio-dir "$AUDIO"

# 4. one campaign per annotator language pair: English plus one target language
for L in en en,de en,cs en,pl en,sk; do
    id="earnings25_merged_${L//,/}"
    python make_pearmut_campaign.py merged_annotations.jsonl --clips-dir clips \
        --lang "$L" --min-models "$MIN_MODELS" \
        --shuffle off --show-model-names on --slim \
        --campaign-id "$id" \
        --copy-assets "${PEARMUT_ROOT:-.}/data/assets" \
        -o "campaign_${id#earnings25_merged_}.json"
done

echo
echo "now load them into pearmut:  pearmut add -o campaign_*.json   then:  pearmut run"
