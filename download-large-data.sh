#!/bin/bash
# Download large data that our scripts/workflow need but that is not committed.
# See README.md for how to add a new section.
set -euo pipefail
cd "$(dirname "$0")"

### earnings25-asr-to-pearmut: pre-cut audio clips (3,104 mp3, ~220 MB)
# One clip per annotated Canary segment, i.e. the assets the Pearmut campaigns refer to.
# Covers the merged set (ASR-harm + DSPy divergencies); the ASR-only campaign uses a subset.
# Only needed if you do NOT have the 3.1 GB earnings-25 testset-segmented locally; with it,
#   cd earnings25-asr-to-pearmut && python cut_clips.py annotations.jsonl clips \
#       --audio-dir ../../earnings25_raw/earnings-25/testset-segmented/audio
# regenerates them in ~2 minutes.
#
# TODO: attach earnings25-asr-to-pearmut/clips.tar.gz to a release
#       (https://github.com/Gldkslfmsd/nmee/releases/new) and put the URL here.
CLIPS_URL="${CLIPS_URL:-}"
if [ -z "$CLIPS_URL" ]; then
    echo "skipping pearmut clips: CLIPS_URL not set yet (see the TODO in this script)" >&2
else
    mkdir -p earnings25-asr-to-pearmut
    wget -O - "$CLIPS_URL" | tar -xz -C earnings25-asr-to-pearmut
    echo "clips extracted to earnings25-asr-to-pearmut/clips/"
fi
