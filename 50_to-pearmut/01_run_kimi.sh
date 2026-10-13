#!/bin/bash

outdir=out-Kimi
mkdir -p $outdir

cmd() {
    l=$1
    python3 make_pearmut_campaign.py  \
        ../40_preselect/out/Kimi_$l.jsonl  -o $outdir/campaign_Kimi_$l.json --campaign-id earnings-kimi-$l \
        --users 2 --assets-url ./assets/earnings-25/ --seconds-per-page 90 --lang $l
}

cmd cs
cmd de
cmd fr
