#!/bin/bash
for x in GLM Kimi gpt-oss-120B ; do
    python3 ../make_pearmut_campaign.py llm-outputs/$x""_cs.jsonl -o campaigns/campaign_$x.json --seconds-per-page 0 --campaign-id $x""_cs --assets-url ./assets/earnings-25
done
