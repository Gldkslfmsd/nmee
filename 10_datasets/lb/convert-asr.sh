#!/bin/bash
mkdir -p outputs
for  x in nmee_lb/asr_w_timestamps/*json ; do 
    o=$(basename $x)
    o=${o/luxasr_/}    
    o=${o/.json/}
    o=outputs/$o"".lb.jsonl
    python3 convert_to_jsonl.py $x $o &
done
wait
