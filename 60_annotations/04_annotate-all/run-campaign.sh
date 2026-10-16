#!/bin/bash
x=dummy
mkdir -p campaigns
python3 ../../50_to-pearmut/make_pearmut_campaign.py head*jsonl -o campaigns/campaign_$x.json --seconds-per-page 90 --campaign-id $x""_cs --assets-url ./assets/earnings-25 --no-asr-column --lang cs --no-info
