Ondrej Bojar (and Claude) implemented here the search for ASR/SLT errors using LLM accessible via an API and using the DSPy wrapper around LLMs.

The scripts are numbered to indicate the pipeline:

10-rough-align.py processes segmented Earnings-2025, merging our ASR and SLT target languages into joint files.

20-find-bad-translation-divergencies.py runs DSPy over the joint files, using LLMs to spot divergencies

30-merge-annotations.py merges several LLM outputs into compact files (no LLMs needed for this)

The last step ./merged_errors_to_pearmut.py will probably still go outside of this directory, so not numbering it. It converts the merged files to Pearmut campaing and creates the sound file snippets needed as data/assets/.
