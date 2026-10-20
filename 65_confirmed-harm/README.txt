python3 10_filter_harm.py ../30_segment-align/earnings-systems/earnings.fr.jsonl --confirmed confirmed.fr.jsonl -o harm.fr.jsonrm.fr.jsonl

confirmed.fr.jsonl
- confirmed harmful errors, from ../60_annotations

../30_segment-align/earnings-systems/earnings.fr.jsonl
- all system outputs, including new ones, aligned

harm.fr.jsonl
- systems outputs, including new ones, filtered for only the confirmed segments
