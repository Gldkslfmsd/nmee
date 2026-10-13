#!/usr/bin/env python3
import sys
import json

out = [open(f+".jsonl","w") for f in sys.argv[1:]]

def load_parts(p):
    with open(p,"r") as f:
        return list(x.strip() for x in f)

parts = [load_parts(p) for p in sys.argv[1:]]

data = {}
for line in sys.stdin:
    j = json.loads(line)
    d = j["document"]
    if d not in data:
        data[d] = []
    data[d].append(line)

for part,out_f in zip(parts,out):
    for p in part:
        if p not in data: continue
        for l in data[p]:
            out_f.write(l)

for f in out:
    f.close()
