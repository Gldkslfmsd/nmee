#!/bin/bash
# Runs 20-find-bad-translation-divergencies.py over all ./roughaligned/*.json
# files, once per model, one model after another (sequentially), so the
# e-INFRA gateway's per-key in-flight-request cap isn't shared across models
# at the same time. Each model's own run still parallelizes across files
# (see --concurrency in the python script).
set -euo pipefail
cd "$(dirname "$0")"

: "${E_INFRA_API_TOKEN:?Set E_INFRA_API_TOKEN (or CESNET_API_KEY) before running this script.}"

MODELS=("gpt-oss-120B" "GLM" "Kimi" "DeepSeek")

for MODEL in "${MODELS[@]}"; do
    echo "=== $(date -Is) starting model: $MODEL ==="
    ./venv/bin/python 20-find-bad-translation-divergencies.py --model "$MODEL" 15-roughaligned/*.json
    echo "=== $(date -Is) finished model: $MODEL ==="
done

echo "=== $(date -Is) all models done ==="
