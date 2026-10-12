#!/bin/bash
mkdir -p outputs
for l in en de fr ; do
    for  x in nmee_lb/mt_data/mt_$l/*jsonl ; do 
        o=$(basename $x)
        o=${o/_segmented_gemma-4-9-combined.jsonl/}    
        o=outputs/$o"".$l.jsonl
        python3 convert_mt_to_jsonl.py $x $o
    done
done
