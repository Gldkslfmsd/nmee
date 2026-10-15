#!/bin/bash -v

for model in Kimi GLM ; do
    INPUT=../../40_preselect/out-lb/$model""_en+de+fr.jsonl


    if [ ! -f order.tsv ]; then
        echo first, order all the chunks. Create order.tsv >&2
        exit 1
    fi

    N=ordered.$(basename $INPUT)
    python3 chunk_order.py $INPUT --chunk-order order.tsv -o $N

    mkdir -p campaigns
    for lan in en,de,fr en de fr en,de en,fr de,fr ; do
        campaign=luxparl-$model""_$lan-14.10.
        out=campaigns/campaign_$campaign""json
        ../../p3.12/bin/python3 ../make_pearmut_campaign.py $N \
            -o $out --campaign-id $campaign \
             --assets-url ./assets/luxparl/ --seconds-per-page 40 --lang $lan --include-asr if_flagged \
            --users 3 --usernames nr
    done
done
