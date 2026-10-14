Here is a campaign from LuxParl (lb->en+de+fr)
- two campaign, annotated by Kimi and GLM. 
- we don't know which is better. Suggestion: let's start with Kimi, if immediately spotted low precision, let's try GLM. If still low precision, let's annotate some pages, and then let's improve (LLM prompt, post-selection, ...)


chunk_order.py
- it virtually chunks audio to 20-minute chunks, shuffles them, and writes the order to order.tsv
order.tsv
- the order, shuffled once, more LLM annotations can be put into this order
- created by:

python3 chunk_order.py ../../30_segment-align/luxparl-segmented/luxparl-merged.jsonl --chunk-order order.tsv -o order.luxparl-merged.jsonl



20_lb.sh
- includes chunk_order
- and then make campaign generation 

campaign_luxparl-GLM_en+de+fr-14.10.json
campaign_luxparl-Kimi_en+de+fr-14.10.json

ordered.GLM_en+de+fr.jsonl
ordered.Kimi_en+de+fr.jsonl
- 
