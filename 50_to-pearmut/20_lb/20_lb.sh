#!/bin/bash -v

for model in Kimi GLM ; do
    INPUT=../../40_preselect/out-lb/$model""_en+de+fr.jsonl

    campaign=luxparl-$model""_en+de+fr-14.10.
    out=campaign_$campaign""json

    if [ ! -f order.tsv ]; then
        echo first, order all the chunks. Create order.tsv >&2
        exit 1
    fi

    N=ordered.$(basename $INPUT)
    python3 chunk_order.py $INPUT --chunk-order order.tsv -o $N

    ../../p3.12/bin/python3 ../make_pearmut_campaign.py $N \
        -o $out --campaign-id $campaign \
         --assets-url ./assets/luxparl/ --seconds-per-page 40 --lang en,de,fr --include-asr if_flagged \
        --users 3 --usernames nr
done
