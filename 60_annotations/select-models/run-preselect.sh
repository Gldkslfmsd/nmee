#!/bin/bash

LANGUAGES=("cs")
MODELS=("gpt-oss-120B" "GLM" "Kimi")
BACKEND="einfra"

OUTDIR=llm-outputs
#INPUT="out3.jsonl"
#head -n 2000 ../30_segment-align/canary-skimmed.jsonl > $INPUT
INPUT=input.jsonl

mkdir -p $OUTDIR

for MODEL in "${MODELS[@]}"; do
    echo "==============================================="
    echo "Starting with model: $MODEL"
    echo "==============================================="
    echo
    
    for LANG in "${LANGUAGES[@]}"; do
        SYSTEM="canary_${LANG}"
        OUTPUT="$OUTDIR/${MODEL}_${LANG}.jsonl"
        
        echo "[$(date +'%H:%M:%S')] Running $MODEL for ${LANG} (${SYSTEM})"
        python3 ../../40_preselect/40_find_harmful_errors.py \
            --backend "$BACKEND" \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --model "$MODEL" \
            --show canary_asr "$SYSTEM" gold_transcript \
            --annotate canary_asr "$SYSTEM" \
            --batch-size 1 \
            --group-size 100 \
            --resume
        
        echo "✓ Finished ${LANG}; output: $OUTPUT"
        echo
    done
    
    echo "==============================================="
    echo "Finished all languages with $MODEL"
    echo "==============================================="
    echo
done

echo "All done. Outputs:"
ls -lh o_*.jsonl
