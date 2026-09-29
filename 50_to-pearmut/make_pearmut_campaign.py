#!/usr/bin/env python3
"""Build a Pearmut campaign from flagged harmful errors (the custom NMEE protocol).
Merges the IWSLT and the Earnings25 campaign builders. Input is one JSONL from the LLM annotation
(40_find_harmful_errors.py) or from any other flagger, one line per segment with at least one flagged
span:
{"document", "dataset", "src_language", "audio", "beg", "end", "segmented_by",
 "asr": "...", "asr_system": "canary_asr", "gold_transcript": "...",
 "targets": [{"tgt_lan", "system", "text", "span", "span_start", "span_end", "intended",
              "harm_types", "harmfulness", "explanation", "error_source", ...}],
 "annotator": {...}}

Only the fields Pearmut actually needs are written to the campaign: per item "item_id", "src",
"tgt", "instructions" and (unless --no-prefill) "error_spans". Columns of one item: the ASR
transcript first (--no-asr-column turns it off), then the target systems that have a flagged span
in this segment, at most --max-systems of them (0 = all). The clip and the gold transcript are the
source side; every flagged span is described in the item's instructions (as an HTML table, see
instruction_html.py) and, unless --no-prefill, pre-highlighted in its column. Columns are labelled
with the system name (--show-model-names) and not shuffled.

Reference translations come from a separate file: --references-file is the aligned JSONL of
30_add_and_align_sentences.py (with "text" per system), and --references names the systems in it that
hold references, e.g. --references reference_cs reference_de. Segments are matched by document and
start time.

Usage:
python make_pearmut_campaign.py harmful.jsonl --clips-dir .. \\
    --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json
pearmut add -o campaign.json
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from instruction_html import instruction_html, esc

MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
        ".m4a": "audio/mp4", ".opus": "audio/ogg"}


# ---------------------------------------------------------------- references

def load_references(path, fields):
    """{(document, round(beg, 2)): {system: text}} from the aligned JSONL."""
    refs = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        picked = {f: r.get("text", {})[f] for f in fields if r.get("text", {}).get(f)}
        if picked:
            refs[(r.get("document"), round(float(r.get("beg", 0)), 2))] = picked
    return refs


def find_reference(refs, rec, tolerance=0.05):
    doc, beg = rec.get("document"), round(float(rec.get("beg", 0)), 2)
    hit = refs.get((doc, beg))
    if hit is not None:
        return hit
    for (d, b), v in refs.items():  # small timestamp drift
        if d == doc and abs(b - beg) <= tolerance:
            return v
    return {}


# ---------------------------------------------------------------- selecting the flagged spans

def select(targets, args):
    """Filter the flagged spans of one segment."""
    out = targets
    if args.harmfulness is not None:
        out = [t for t in out if t.get("harmfulness") is None or t["harmfulness"] >= args.harmfulness]
    if args.langs:
        out = [t for t in out if t.get("tgt_lan") in args.langs]
    return out


def columns(rec, args):
    """(tgt, spans): the ASR transcript first, then the target systems that have a flagged span."""
    tgt, spans = {}, defaultdict(list)
    asr_system = rec.get("asr_system") or "asr"
    if rec.get("asr") and not args.no_asr_column:
        tgt[asr_system] = rec["asr"]

    order = []
    for t in rec["targets"]:
        if t["system"] not in order:
            order.append(t["system"])
    if args.max_systems:
        keep = [s for s in order if s != asr_system][: args.max_systems]
        order = [s for s in order if s == asr_system or s in keep]

    for t in rec["targets"]:
        system = t["system"]
        if system not in order or (system == asr_system and args.no_asr_column):
            continue
        if tgt.setdefault(system, t.get("text", "")) != t.get("text", ""):
            print(f"WARNING: {rec.get('document')} @{rec.get('beg')}: system {system} has different "
                  f"texts, keeping the first", file=sys.stderr)
        text = tgt[system]
        if not t.get("span") or t.get("span_start") is None:
            continue  # deletion or unlocalised flag: it stays in the instructions only
        s0 = (t["span_start"] if text[t["span_start"]:t.get("span_end", 0)] == t["span"]
              else text.find(t["span"]))
        if s0 < 0:
            print(f"WARNING: span {t['span']!r} not in {rec.get('document')} @{rec.get('beg')} "
                  f"({system})", file=sys.stderr)
            continue
        # end_i is inclusive; the NMEE protocol has no severity buttons, so severity stays unset
        spans[system].append({"start_i": s0, "end_i": s0 + len(t["span"]) - 1,
                              "severity": args.prefill_severity, "category": None})
    return tgt, spans


def build_item(rec, args, refs, assets_url, stats):
    if not rec.get("audio"):
        print(f"WARNING: {rec.get('document')}: no \"audio\", skipped", file=sys.stderr)
        return None
    audio = Path(rec["audio"])
    rel = Path(*audio.parts[-2:]) if len(audio.parts) > 1 else Path(audio.name)
    clip = Path(args.clips_dir) / audio if args.clips_dir and not audio.is_absolute() else audio
    if not clip.exists():
        stats["missing clips"] += 1
    elif args.copy_assets:
        dst = Path(args.copy_assets) / args.campaign_id / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip, dst)

    src = (f'<audio controls src="{assets_url}/{rel.as_posix()}" '
           f'type="{MIME.get(audio.suffix.lower(), "audio/wav")}"></audio>')
    if rec.get("gold_transcript") and not args.no_gold:
        src += f"<br><i>Gold transcript:</i><br>{esc(rec['gold_transcript'])}"
    for field, text in (find_reference(refs, rec).items() if refs else []):
        src += f"<br><i>Reference ({esc(field)}):</i><br>{esc(text)}"

    tgt, spans = columns(rec, args)
    if not tgt:
        return None

    # only the fields Pearmut consumes; anything else would just bloat the logs
    item = {
        "item_id": f"{rec.get('document')}#{audio.stem.split('.')[-1]}",
        "src": src,
        "tgt": tgt,
        "instructions": instruction_html(rec, rec["targets"]),
    }
    if spans and not args.no_prefill:
        item["error_spans"] = dict(spans)
    stats["items"] += 1
    stats["spans"] += sum(len(v) for v in spans.values())
    return item

# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="JSONL of flagged harmful errors")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--clips-dir", help="directory the \"audio\" paths are relative to (default: as given)")
    ap.add_argument("--campaign-id", default="harm_review")
    ap.add_argument("--assets-url", help="URL prefix for clips (default: ./assets/<campaign-id>)")
    ap.add_argument("--copy-assets", metavar="ASSETS_DIR",
                    help="copy the needed clips to ASSETS_DIR/<campaign-id>/")

    # what to keep
    ap.add_argument("--harmfulness", type=int, choices=range(0, 6), metavar="0-5",
                    help="keep only spans with harmfulness >= this (spans without one are kept)")
    ap.add_argument("--lang", help="comma-separated target languages to keep, e.g. cs or cs,de "
                                   "(default: all)")

    # columns
    ap.add_argument("--max-systems", type=int, default=2,
                    help="how many target systems per item, in the order they are flagged "
                         "(default: 2; 0 = all)")
    ap.add_argument("--no-asr-column", action="store_true",
                    help="do not show the ASR transcript as the first column")
    ap.add_argument("--no-gold", action="store_true", help="do not show the gold transcript")
    ap.add_argument("--references-file", help="aligned JSONL (30_add_and_align_sentences.py) holding "
                                              "the reference translations in \"text\"")
    ap.add_argument("--references", nargs="+", default=[],
                    help="system names in --references-file to show as references, e.g. reference_cs")

    # pre-filled spans
    ap.add_argument("--no-prefill", action="store_true", help="do not pre-highlight the flagged spans")
    ap.add_argument("--prefill-severity", default=None,
                    help="severity stored in the pre-filled spans (default: none; the NMEE protocol "
                         "has no severity buttons)")

    # campaign
    ap.add_argument("--template", default="../custom_nmee_demo.json",
                    help="campaign JSON whose \"info\" block is copied verbatim "
                         "(default: custom_nmee_demo.json next to this script)")
    ap.add_argument("--users", type=int, default=1, help="number of annotator tasks (default: 1)")
    ap.add_argument("--partition", action="store_true",
                    help="split the documents across --users instead of giving each user all of them")
    ap.add_argument("--shuffle", choices=["keep", "on", "off"], default="off",
                    help="override the template's model shuffling; the columns are systems, so "
                         "shuffling them only confuses the annotator (default: off)")
    ap.add_argument("--show-model-names", choices=["keep", "on", "off"], default="on",
                    help="label each column with its system name (default: on)")
    args = ap.parse_args()

    if not args.template:
        sys.exit("no --template given and no custom_nmee_demo.json found next to this script")
    args.langs = set(args.lang.split(",")) if args.lang else None
    assets_url = (args.assets_url or f"./assets/{args.campaign_id}").rstrip("/")
    refs = (load_references(args.references_file, args.references)
            if args.references_file and args.references else None)

    by_doc, n_errors = defaultdict(list), 0
    for line in open(args.annotations, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["targets"] = select(rec.get("targets", []), args)
        if rec["targets"]:
            by_doc[rec.get("document")].append(rec)
            n_errors += len(rec["targets"])

    stats, documents = defaultdict(int), []
    for doc in sorted(by_doc):
        items = [it for it in (build_item(r, args, refs, assets_url, stats)
                                for r in sorted(by_doc[doc], key=lambda r: r.get("beg") or 0)) if it]
        if items:
            documents.append(items)

    if args.partition:
        tasks = [t for t in (documents[i::args.users] for i in range(args.users)) if t]
    else:
        tasks = [documents for _ in range(args.users)]

    info = json.load(open(args.template, encoding="utf-8"))["info"]
    if args.shuffle != "keep":
        info["shuffle"] = args.shuffle == "on"
    if args.show_model_names != "keep":
        info["show_model_names"] = args.show_model_names == "on"

    json.dump({"info": info, "campaign_id": args.campaign_id, "data": tasks},
              open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if args.copy_assets:
        print(f"clips copied to {Path(args.copy_assets).resolve() / args.campaign_id}", file=sys.stderr)
        print(f"clips referenced as {assets_url}/<document>/<clip>", file=sys.stderr)
    if stats["missing clips"]:
        print(f"WARNING: {stats['missing clips']} clip(s) not found (--clips-dir {args.clips_dir})",
              file=sys.stderr)
    print(f"{len(documents)} documents, {stats['items']} items, {n_errors} flagged span(s), "
          f"{stats['spans']} pre-filled, {len(tasks)} task(s) "
          f"{'partitioned' if args.partition else 'replicated'} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()