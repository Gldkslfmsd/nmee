import json
import sys
for line in sys.stdin:
    line = line.strip()
    j = json.loads(line)

    del j["words"]
    del j["alignment"]
    del j["gold_meta"]
    del j["segmented_by"]

    o = json.dumps(j, ensure_ascii=False)
    print(o)
