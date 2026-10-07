#!/bin/bash
for x in GLM Kimi gpt-oss-120B ; do
    python3 ../make_pearmut_campaign.py ../../40_preselect/out3/$x""_cs.jsonl -o campaign_$x.json --seconds-per-page 0 --campaign-id $x""_cs --assets-url ./assets/earnings-25
done
