this is experiment with annotations GLM vs. gpt-oss vs. Kimi, first 3 documents earnings-25 en-cs 
- I selected first 2000 segments of earnings
    - input.jsonl
- I processed ../40_preselect/ ...
    - run-preselect.sh
    - the outputs are in llm-outputs/
- then converted them into pearmut campaign under campaigns/
    - with ./run-campaign.sh
    - campaings/ dir
- then, add to pearmut: pearmut add campaings/*json
    - then, symlink earnings-25 audio into data/assets/... , 
- then, run pearmut
- then annotated -> data/annotations
    - I made sure to be consistent, I looked at all at once and made the same decisions

- then, after 3 documents, I committed, and evaluated GLM vs. gtp-oss vs. Kimi
    - then saved -> *jsonl
    - then evaluated: *txt
        - read them and interpret results
    - I found that gpt-oss is bad. I'm going to annotate more documents GLM vs. Kimi
