#!/bin/bash -v

outdir=10_earnings-kimi-13.10./
out=$outdir/campaigns
mkdir -p $out




cmd_cs() {
    l=cs
    INPUT=$1
    part=$2
    if [ "$part" = p2 ]; then
        partition="--users 5 --partition --usernames vh "
    else
        partition="--users 6 --usernames vh dm mk dj ob"
    fi
    python3 make_pearmut_campaign.py $INPUT \
        -o $out/earnings-kimi-$l-13.10.$part.json --campaign-id earnings-kimi-$l-13.10.$part \
         --assets-url ./assets/earnings-25/ --seconds-per-page 90 --lang $l --include-asr if_flagged \
         $partition
}


cmd() {
    l=$1
    INPUT=$2
    part=$3
    if [ "$part" = p2 ]; then
        partition="--users 3 --partition"
    else
        partition="--users 3 "
    fi
    python3 make_pearmut_campaign.py $INPUT \
        -o $out/earnings-kimi-$l-13.10.$part.json --campaign-id earnings-kimi-$l-13.10.$part \
         --assets-url ./assets/earnings-25/ --seconds-per-page 90 --lang $l --include-asr if_flagged \
         $partition
}

#cmd cs $outdir/doc-order.part1.jsonl p1
#cmd cs $outdir/doc-order.part2.jsonl p2

for lan in fr pl ; do
    d=$outdir/inputs-$lan
    mkdir -p $d
    cd $d
    cp ../doc-order.part{1,2} .
    pwd
    python3 ../../split-docs.py doc-order.part1 doc-order.part2 < ../../../40_preselect/out/Kimi_$lan.jsonl
    cd ../..
    cmd $lan $outdir/inputs-$lan/doc-order.part1.jsonl p1   
    cmd $lan $outdir/inputs-$lan/doc-order.part2.jsonl p2
done
