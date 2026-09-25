#!/usr/bin/env python3
"""Add the Earnings-25 gold transcripts to the sentence-level JSONL of 30_add_and_align_sentences.py.

--gold is the dataset's JSONL, one document per line, with "id" (the document), "transcript" (the whole
gold text, no timestamps) and metadata. Each transcript is re-segmented into the document's existing
sentences with mweralign (--align mwer, aligned to a system in the same language, by default the first
one matching --lan), exactly as 30_add_and_align_sentences.py does it, and stored as a system
(--name, default gold_transcript, is_human = true).

Every line of a document also gets:
  "gold_meta": {"participants": ["Operator", "Matthew Fort", ...],   # from participants, else from
                                                                    # speaker_attributions
                "extra_fields": {...}}                              # Company, Country, Industry, ...

Documents of --gold that are not in --input are skipped (they are simply other documents of the
dataset); documents of --input that are not in --gold are reported.

Usage:
    python 35_add_earnings_gold.py --input aligned/all.jsonl --output aligned/all+gold.jsonl \\
        --gold earnings25/metadata.jsonl --lan en --align-to canary_asr
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_sibling(pattern):
    """Import the add_and_align_sentences script next to this file (its name starts with a digit, so
    it cannot be imported by name)."""
    here = Path(__file__).resolve().parent
    matches = sorted(here.glob(pattern))
    if not matches:
        sys.exit(f"{pattern} not found next to {Path(__file__).name}")
    spec = importlib.util.spec_from_file_location("add_and_align_sentences", matches[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AAS = load_sibling("*add_and_align_sentences.py")


def gold_meta(rec):
    """The participants and the remaining extra_fields of an Earnings-25 line. The speaker attributions
    are only used as a fallback source of names (their values are the participants); everything else
    (audio path, duration, language) is already elsewhere in the pipeline."""
    extra = dict(rec.get("extra_fields") or {})
    attributions = extra.pop("speaker_attributions", None)
    participants = extra.pop("participants", None)
    if isinstance(participants, str):
        participants = [p.strip() for p in participants.split(",") if p.strip()]
    if not participants and attributions:
        if isinstance(attributions, str):
            try:
                attributions = json.loads(attributions)
            except json.JSONDecodeError:
                attributions = None
        if isinstance(attributions, dict):
            participants = list(dict.fromkeys(attributions.values()))
    meta = {"participants": participants, "extra_fields": extra}
    return {k: v for k, v in meta.items() if v not in (None, {}, [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="output of 30_add_and_align_sentences.py")
    ap.add_argument("--output", help="output JSONL (default: edit --input in place)")
    ap.add_argument("--gold", required=True, nargs="+", help="Earnings-25 JSONL file(s) with id/transcript")
    ap.add_argument("--name", default="gold_transcript", help="system name (default: gold_transcript)")
    ap.add_argument("--lan", default="en", help="language of the transcripts (default: en)")
    ap.add_argument("--align-to", help="system to align to (default: the first one with --lan)")
    ap.add_argument("--align", choices=["mwer"], default="mwer", help="only mwer makes sense here")
    ap.add_argument("--mwer-tokenizer", default="spm32k", help="mweralign tokenizer (default spm32k)")
    ap.add_argument("--mweralign-dir", help="path to a mweralign checkout (default: third_party/mweralign)")
    ap.add_argument("--joiner", default=" ")
    ap.add_argument("--overwrite", action="store_true", help="replace the system if already present")
    ap.add_argument("--with-words", action="store_true", help="store the words (no timestamps here)")
    ap.add_argument("--no-meta", action="store_true", help="do not add \"gold_meta\"")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="report a failing document and go on")
    args = ap.parse_args()
    args.is_human = True  # a gold transcript is human by definition
    args.src_language = args.lan  # used only when deriving document IDs from file names

    recs = AAS.read_jsonl(args.input)
    if not recs or not AAS.is_aligned_format(recs):
        sys.exit(f"{args.input}: not the output of 30_add_and_align_sentences.py")
    groups = AAS.group_documents(recs, args)
    print(f"{len(groups)} document(s) in {args.input}", file=sys.stderr)

    gold = {}
    for path in args.gold:
        for g in AAS.read_jsonl(path):
            doc = g.get("id")
            if doc in gold:
                sys.exit(f"--gold: document {doc!r} appears twice")
            gold[doc] = g
    missing = [d for d in groups if d not in gold]
    if missing:
        print(f"WARNING: {len(missing)} document(s) of --input are not in --gold: "
              f"{', '.join(map(str, missing[:5]))}{' ...' if len(missing) > 5 else ''}", file=sys.stderr)

    todo = [d for d in groups if d in gold]
    failed = []
    for n, doc in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {doc}", file=sys.stderr)
        g = gold[doc]
        text = str(g.get("transcript") or "").strip()
        if not text:
            print(f"FAILED {doc}: empty transcript", file=sys.stderr)
            failed.append(doc)
            continue
        words = [{"word": t} for t in text.split()]
        try:
            AAS.add_system(groups[doc], words, [list(range(len(words)))], args)
        except Exception as e:
            if not args.continue_on_error:
                raise
            print(f"FAILED {doc}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(doc)
            continue
        if not args.no_meta:
            meta = gold_meta(g)
            for r in groups[doc]:
                r["gold_meta"] = meta

    out_path = args.output or args.input
    AAS.write_jsonl(out_path, recs)
    print(f"\n-> {out_path}: {len(recs)} sentences in {len(groups)} document(s), "
          f"{len(todo) - len(failed)} with a gold transcript, systems: "
          f"{', '.join(recs[0]['systems_info'])}", file=sys.stderr)
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()