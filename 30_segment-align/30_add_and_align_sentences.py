#!/usr/bin/env python3
"""Build and extend a sentence-level, multi-system JSONL.

The JSONL may hold one document or many merged together (each line carries its "document"); documents
are handled separately but read and written as one file.

Create (once per document): read the sentence segmentation made by asr_sentences.py
(one sentence per line: {"segment", "start", "end", "words"}). Its text becomes the first system
(--name, e.g. canary_asr).

Add (once per system): read the output of this script and add one more system (--add), split into the
existing sentences by one of three methods (--align mwer | vecalign | time; see alignments.py).

--add takes one file, or many: several files, a glob, or a directory. All of them are handled in this
one process, so the LaBSE / SentencePiece models are loaded once -- the point with a few hundred
documents. Each --add file is paired with a document of --input by its name (DOC.jsonl, DOC.txt,
DOC.<lan>.*); a file whose document is not in --input is an error, and documents of --input without an
--add file keep their systems unchanged (with a warning).

--add accepts Canary-style JSONL ("words" with "start"/"end" per line), word-per-line JSONL, a NeMo
hypothesis JSON, or plain text (e.g. a gold transcript or reference; no timestamps, so not usable with
--align time).

Output: one sentence per line:
  {"document", "dataset", "src_language",
   "systems_info": {"canary_asr": {"lan": "en", "is_human": false}, ...},
   "segmented_by": "canary_asr",
   "audio": "<audio-dir>/<doc>/<doc>.<NNNN>.wav",     # clip made by segment_audio.py
   "beg": 12.3, "end": 17.8,                            # seconds in the original long audio
   "text": {"canary_asr": "...", "canary_mt_cs": "..."},
   "alignment": {"canary_mt_cs": "vecalign 1:1", ...},  # how each added system was aligned here
   "words": {"canary_asr": [{"word", "start", "end"}, ...], ...}}   # only with --with-words

Alignment labels: "mwer", "time", or "vecalign a:b" where a existing sentences were aligned to b added
sentences; "split" means the added sentence(s) of an n:m unit were divided among the n existing
sentences (existing sentences are never merged); "+ins" means unaligned added sentences were attached
to this sentence; "1:0" means nothing was aligned to this sentence.

Usage:
    # create
    python add_and_align_sentences.py --input sentences/DOC.en.jsonl --output aligned/DOC.jsonl \\
        --dataset earnings25 --src-language en --name canary_asr --lan en --audio-dir segments
    # merge the documents into one file, then add a system to all of them in one process
    cat aligned/*.jsonl > aligned/all.jsonl
    python add_and_align_sentences.py --input aligned/all.jsonl --output aligned/all+mt.jsonl \\
        --add mt/cs/ --name canary_mt_cs --lan cs --align vecalign
    # add a gold transcript or a reference (same language as the system it is aligned to)
    python add_and_align_sentences.py --input aligned/DOC.jsonl \\
        --add gold/DOC.txt --name gold_transcript --lan en --is-human --align mwer

--name, --lan and --is-human describe the system of the input sentences when creating, and the added
system when adding. Creating and adding are separate calls.

Metadata arguments (--document, --dataset, --src-language, --segmented-by, --audio-dir) are optional
when the input is already aligned: fields whose argument is given are updated, all others are kept as
they are. When creating, fields whose argument is not given are set to null (only --name is required).
The document ID, if neither given nor already in the input, is taken from the input file name:
DOC.jsonl, or DOC.<lan>.jsonl where <lan> is --lan or --src-language.
"""
import argparse
import glob as globmod
import json
import sys
import time
from pathlib import Path

from alignments import Options, align, has_times


# ---------------------------------------------------------------- I/O

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_add(path):
    """Return (words, segments): words = [{"word", "start"?, "end"?}], segments = list of word-index
    lists (the file's own lines/segments). Plain text gives words without timestamps."""
    text = Path(path).read_text(encoding="utf-8")
    records = None
    try:
        data = json.loads(text)
        records = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        try:
            records = [json.loads(l) for l in text.splitlines() if l.strip()]
        except json.JSONDecodeError:
            records = None
    words, segments = [], []

    def add(ws):
        idx = []
        for w in ws:
            if str(w.get("word", "")).strip():
                idx.append(len(words))
                words.append(w)
        if idx:
            segments.append(idx)

    if records is None or not all(isinstance(r, dict) for r in records):  # plain text
        for line in text.splitlines():
            add([{"word": t} for t in line.split()])
        return words, segments
    for r in records:
        if "words" in r:
            add(r["words"])
        elif isinstance(r.get("timestamp"), dict) and "word" in r["timestamp"]:
            add(r["timestamp"]["word"])
        elif "word" in r:
            add([r])
        elif "segment" in r or "text" in r:
            add([{"word": t} for t in str(r.get("segment", r.get("text", ""))).split()])
    return words, segments


def is_aligned_format(recs):
    return bool(recs) and isinstance(recs[0].get("text"), dict) and "systems_info" in recs[0]


# ---------------------------------------------------------------- document ID, metadata

def stem_doc(path, args):
    """Document ID from a file name: DOC.jsonl, DOC.txt, or DOC.<lan>.* with <lan> = --lan or
    --src-language."""
    name = Path(path).name
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else Path(name).stem
    for lan in (args.lan, args.src_language):
        if lan and stem.endswith(f".{lan}"):
            return stem[: -len(lan) - 1]
    return stem


def set_audio(recs, audio_dir):
    missing = 0
    for i, r in enumerate(recs):
        doc = r["document"]
        if not doc:
            sys.exit("--audio-dir needs the document ID (--document, or \"document\" in the input)")
        r["audio"] = str(Path(audio_dir) / doc / f"{doc}.{i:04d}.wav")
        missing += not Path(r["audio"]).exists()
    if missing:
        print(f"WARNING: {missing}/{len(recs)} audio clips not found under {audio_dir} "
              f"(run segment_audio.py on the same sentence files)", file=sys.stderr)


def create(segs, in_path, args):
    if args.name is None:
        sys.exit("--name is required when the input is asr_sentences.py output")
    document = args.document or stem_doc(in_path, args)
    if not args.document:
        print(f"note: document ID taken from the input file name: {document!r}", file=sys.stderr)
    unset = [a for a in ("dataset", "src_language", "lan") if getattr(args, a) is None]
    if unset:
        print(f"note: not given, set to null: {', '.join('--' + a.replace('_', '-') for a in unset)}",
              file=sys.stderr)
    out = []
    for s in segs:
        rec = {
            "document": document,
            "dataset": args.dataset,
            "src_language": args.src_language,
            "systems_info": {args.name: {"lan": args.lan, "is_human": args.is_human}},
            "segmented_by": args.segmented_by or args.name,
            "audio": None,
            "beg": s["start"],
            "end": s["end"],
            "text": {args.name: s.get("segment", "")},
        }
        if args.with_words:
            rec["words"] = {args.name: s.get("words", [])}
        out.append(rec)
    if args.audio_dir:
        set_audio(out, args.audio_dir)
    print(f"created {len(out)} sentences, system {args.name!r}", file=sys.stderr)
    return out


def update_metadata(recs, in_path, args):
    """Aligned input: update only the fields whose argument was given; keep all others."""
    changed = []
    if args.document is None and not recs[0].get("document"):
        doc = stem_doc(in_path, args)
        print(f"note: document ID taken from the input file name: {doc!r}", file=sys.stderr)
        for r in recs:
            r["document"] = doc
        changed.append("document")
    for field in ("document", "dataset", "src_language", "segmented_by"):
        val = getattr(args, field)
        if val is not None:
            for r in recs:
                r[field] = val
            changed.append(field)
    if args.audio_dir:
        set_audio(recs, args.audio_dir)
        changed.append("audio")
    if changed:
        print(f"updated: {', '.join(changed)}", file=sys.stderr)
    return recs


# ---------------------------------------------------------------- documents and the --add files

ADD_SUFFIXES = (".jsonl", ".txt")


def add_suffixes(args):
    if args.add_suffix:
        return [args.add_suffix]
    return ([f".{args.lan}{x}" for x in ADD_SUFFIXES] if args.lan else []) + list(ADD_SUFFIXES)


def expand_add(patterns, args):
    """--add: files, globs and directories -> a list of files (a directory gives every file with a
    known suffix)."""
    out = []
    for p in patterns:
        paths = sorted(globmod.glob(p)) if any(c in p for c in "*?[") and not Path(p).exists() else [p]
        if not paths:
            sys.exit(f"--add: no file matches {p!r}")
        for q in paths:
            if Path(q).is_dir():
                found = [str(f) for suf in add_suffixes(args) for f in sorted(Path(q).glob(f"*{suf}"))]
                if not found:
                    sys.exit(f"--add: no {', '.join(add_suffixes(args))} file in {q}")
                out += found
            elif Path(q).is_file():
                out.append(q)
            else:
                sys.exit(f"--add: no such file: {q}")
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def group_documents(recs, args):
    """{document: [records]}, in the order they appear in the file."""
    groups = {}
    for r in recs:
        groups.setdefault(r.get("document"), []).append(r)
    if len(groups) > 1 and None in groups:
        sys.exit("--input: some lines have no \"document\"; cannot tell the documents apart")
    return groups


def match_add(adds, groups, args):
    """[(document, add path)]; an --add file whose document is not in --input is an error."""
    pairs, missing = [], []
    for a in adds:
        doc = stem_doc(a, args)
        if doc in groups:
            pairs.append((doc, a))
        elif len(groups) == 1 and len(adds) == 1:  # single document, single file: trust the user
            pairs.append((next(iter(groups)), a))
        else:
            missing.append((doc, a))
    if missing:
        show = ", ".join(f"{d} ({Path(a).name})" for d, a in missing[:5])
        sys.exit(f"--add: {len(missing)} file(s) whose document is not in {args.input}: {show}"
                 f"{' ...' if len(missing) > 5 else ''}")
    seen = {d for d, _ in pairs}
    if len(seen) < len(pairs):
        dup = [d for d in seen if sum(1 for x, _ in pairs if x == d) > 1]
        sys.exit(f"--add: several files for document(s) {', '.join(sorted(dup)[:5])}")
    idle = [d for d in groups if d not in seen]
    if idle:
        print(f"WARNING: {len(idle)} document(s) in --input have no --add file and keep their systems "
              f"unchanged: {', '.join(map(str, idle[:5]))}{' ...' if len(idle) > 5 else ''}",
              file=sys.stderr)
    return pairs


# ---------------------------------------------------------------- add a system

def add_system(recs, words, segments, args):
    info = recs[0]["systems_info"]
    if args.name in info and not args.overwrite:
        sys.exit(f"system {args.name!r} is already in the input (use --overwrite)")
    if args.lan is None:
        sys.exit("--lan is required with --add")
    if not words:
        sys.exit("--add: no words in the file")

    if args.align == "mwer":
        ref = args.align_to or next((s for s, v in info.items()
                                     if v.get("lan") == args.lan and s != args.name), None)
        if ref is None:
            sys.exit(f"--align mwer needs a system in the same language ({args.lan}) to align to; "
                     f"give --align-to (systems: {', '.join(info)})")
        if info.get(ref, {}).get("lan") not in (None, args.lan):
            print(f"WARNING: aligning {args.lan} to {ref!r} ({info[ref]['lan']}) with mwer; "
                  f"use vecalign for different languages", file=sys.stderr)
    else:
        ref = args.align_to or recs[0]["segmented_by"]
    if args.align != "time":
        print(f"{args.align}: aligning to {ref!r}", file=sys.stderr)
    sent_of, labels = align(args.align, words, segments, recs, ref, Options.from_args(args))

    per_sent = [[] for _ in recs]
    for w, k in zip(words, sent_of):
        per_sent[k].append(w)
    for r, ws, lab in zip(recs, per_sent, labels):
        r["systems_info"][args.name] = {"lan": args.lan, "is_human": args.is_human}
        r["text"][args.name] = args.joiner.join(str(w["word"]).strip() for w in ws)
        r.setdefault("alignment", {})[args.name] = lab
        if args.with_words:
            r.setdefault("words", {})[args.name] = ws
        elif args.name in r.get("words", {}):
            del r["words"][args.name]  # overwritten system: don't keep its old words
    empty = sum(1 for ws in per_sent if not ws)
    print(f"added {args.name!r}: {len(words)} words -> {len(recs)} sentences, "
          f"{empty} sentence(s) without words", file=sys.stderr)
    return recs


def add_to_document(recs, add_path, args):
    """Align one --add file into the records of one document (updated in place)."""
    print(f"adding {add_path}", file=sys.stderr)
    words, segments = load_add(add_path)
    if args.align == "time" and not has_times(words):
        raise ValueError("--align time needs word timestamps; use --align mwer or vecalign")
    add_system(recs, words, segments, args)


def write_jsonl(path, recs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as o:
        for r in recs:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="asr_sentences.py output, or output of this script (one or many documents "
                         "merged into one file)")
    ap.add_argument("--output", help="output JSONL (default: edit --input in place)")
    # metadata
    ap.add_argument("--document", help="document ID, e.g. the audio file stem")
    ap.add_argument("--dataset")
    ap.add_argument("--src-language")
    ap.add_argument("--segmented-by", help="name of the segmentation (default when creating: --name)")
    ap.add_argument("--audio-dir", help="OUT_DIR of segment_audio.py run on the same sentence files")
    # the system: of the input sentences when creating, of --add when adding
    ap.add_argument("--name", help="system name, e.g. canary_asr or canary_mt_cs")
    ap.add_argument("--lan", help="language of the system's text")
    ap.add_argument("--is-human", action="store_true", help="the system is a human (e.g. gold transcript)")
    # adding
    ap.add_argument("--add", nargs="+",
                    help="file(s) of the system to add: one file, or several files / globs / "
                         "directories, matched to the documents of --input by file name and all "
                         "processed in this one process (models loaded once)")
    ap.add_argument("--add-suffix", help="with a directory in --add: take DOC + this suffix "
                                         "(default: .<lan>.jsonl, .<lan>.txt, .jsonl, .txt)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="with several --add files: report a failing document and go on")
    ap.add_argument("--align", choices=["mwer", "vecalign", "time"], default="vecalign",
                    help="mwer: same language as --align-to; vecalign: a translation; time: word "
                         "timestamps only (default: vecalign)")
    ap.add_argument("--align-to", help="system to align to (mwer: a system in the same language, default "
                                       "the first one with --lan; vecalign: default the segmentation system)")
    ap.add_argument("--joiner", default=" ", help="string between words (default: space; '' for zh/ja)")
    ap.add_argument("--overwrite", action="store_true", help="replace a system that is already present")
    ap.add_argument("--with-words", action="store_true",
                    help="store words (with timestamps if any) per system")
    # alignment tuning; defaults are those of Vecalign and mweralign (see alignments.py)
    g = ap.add_argument_group("alignment tuning")
    g.add_argument("--mwer-tokenizer", default="spm32k",
                   help="mweralign tokenizer: spm32k (default), spm64k, spm128k, spm256k, a "
                        "SentencePiece .model path, 'cj', or 'none'")
    g.add_argument("--tgt-split", choices=["moses", "segments"], default="moses",
                   help="vecalign: split the added text into sentences with Moses (default), or use its "
                        "own lines/segments")
    g.add_argument("--embed-model", default="sentence-transformers/LaBSE",
                   help="vecalign: sentence-transformers model (default LaBSE; 'hash' = testing only)")
    g.add_argument("--embed-device", default=None, help="vecalign: cuda, cuda:0, cpu (default: auto)")
    g.add_argument("--embed-fp16", action="store_true", help="vecalign: half precision on CUDA")
    g.add_argument("--embed-batch-size", type=int, default=256, help="vecalign: encoding batch size")
    g.add_argument("--max-size", type=int, default=4,
                   help="vecalign: max sentences in one unit, e.g. 4 allows 1:3, 2:2, 3:1 (default 4)")
    g.add_argument("--search-buffer-size", type=int, default=5, help="vecalign: Vecalign's default, 5")
    g.add_argument("--max-size-full-dp", type=int, default=300, help="vecalign: Vecalign's default, 300")
    g.add_argument("--del-percentile", type=float, default=0.2,
                   help="vecalign: Vecalign's --del_percentile_frac (default 0.2)")
    g.add_argument("--split-method", choices=["embed", "prior"], default="embed",
                   help="vecalign: how to divide one added sentence among several existing sentences: "
                        "embed = where the pieces match the sentences best (default); prior = by "
                        "timestamps, or proportionally to length without them")
    g.add_argument("--split-window", type=int, default=8,
                   help="vecalign: search cut points within this many words of the first guess")
    g.add_argument("--split-length-weight", type=float, default=0.1,
                   help="vecalign: weight of the length prior when splitting (default 0.1)")
    g.add_argument("--split-punct-bonus", type=float, default=0.03,
                   help="vecalign: bonus for cutting after punctuation (default 0.03)")
    g.add_argument("--vecalign-dir", help="path to a vecalign checkout (default: third_party/vecalign)")
    g.add_argument("--mweralign-dir", help="path to a mweralign checkout (default: third_party/mweralign)")
    args = ap.parse_args()

    if args.add and not args.name:
        sys.exit("--name is required with --add")

    recs = read_jsonl(args.input)
    if not recs:
        sys.exit(f"{args.input} is empty")
    aligned = is_aligned_format(recs)
    if args.add and not aligned:
        sys.exit("create the aligned file first (without --add), then add systems")
    if not aligned:
        recs = create(recs, args.input, args)
        groups = group_documents(recs, args)
    else:
        groups = group_documents(recs, args)
        if len(groups) > 1:
            print(f"{len(groups)} documents in {args.input}", file=sys.stderr)
        for doc, doc_recs in groups.items():
            update_metadata(doc_recs, args.input, args)

    failed = []
    if args.add:
        pairs = match_add(expand_add(args.add, args), groups, args)
        t0 = time.time()
        for n, (doc, add_path) in enumerate(pairs, 1):
            if len(pairs) > 1:
                print(f"\n[{n}/{len(pairs)}] {doc}", file=sys.stderr)
            try:
                add_to_document(groups[doc], add_path, args)
            except Exception as e:
                if len(pairs) == 1 or not args.continue_on_error:
                    raise
                print(f"FAILED {doc}: {type(e).__name__}: {e}", file=sys.stderr)
                failed.append(doc)
        if len(pairs) > 1:
            dt = time.time() - t0
            print(f"\ndone: {len(pairs) - len(failed)}/{len(pairs)} document(s) in {dt:.0f}s "
                  f"({dt / len(pairs):.1f}s each)", file=sys.stderr)

    out_path = args.output or args.input
    write_jsonl(out_path, recs)
    print(f"-> {out_path}: {len(recs)} sentences in {len(groups)} document(s), systems: "
          f"{', '.join(recs[0]['systems_info'])}", file=sys.stderr)
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()