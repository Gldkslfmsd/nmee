this is experiment with annotations GLM vs. gpt-oss vs. Kimi, first 3 documents earnings-25 en-cs 
- I processed ../40_ ...
- then converted them into pearmut campaign under campaigns/
- then annotated -> data/annotations
    - I made sure to be consistent, I looked at all at once and made the same decisions 
- then saved -> *jsonl
- then evaluated: *txt

Result:
- gpt-oss is the worst, only 50% precision
- kimi vs. glm is a small difference
- but glm is cheaper!
