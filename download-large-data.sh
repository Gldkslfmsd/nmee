#!/bin/bash
# Download large data that our scripts/workflow need but that is not committed.
# See README.md for how to add a new section.
set -euo pipefail
cd "$(dirname "$0")"

# earnings25 from Zenodo (by Dominik):
if [ ! -f earnings25.ok ]; then  
    wget https://zenodo.org/records/18762168/files/earnings25.zip -O earnings25.zip && unzip earnings25.zip && touch earnings25.ok
else
    echo "earnings25 is already ok. We know it because earnings25.ok exists." >&2
fi

# Canary ST outputs (by Dominik):
if [ "$USER" != machacek ]; then
    # to be done: 
    # download the whole output dir from
    # https://ufallab.ms.mff.cuni.cz/~machacek/mtm26/outputs/
fi

# IWSLT26 cs dev+test (by Dominik):
# to be done


# by Augustin

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
