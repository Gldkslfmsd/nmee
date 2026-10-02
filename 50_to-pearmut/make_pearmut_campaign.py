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
"tgt", "instructions" and (with --prefill) "error_spans". Columns of one item: the ASR
transcript first (--no-asr-column turns it off), then the target systems that have a flagged span
in this segment, at most --max-systems of them (0 = all). The clip and the gold transcript are the
source side; every flagged span is described in the item's instructions (as an HTML table, see
instruction_html.py) and, with --prefill (for view-only debugging), pre-highlighted in its column. Columns are labelled
with the system name (--show-model-names) and not shuffled.

Reference translations come from a separate file: --references-file is the aligned JSONL of
30_add_and_align_sentences.py (with "text" per system), and --references names the systems in it that
hold references, e.g. --references reference_cs reference_de. Segments are matched by document and
start time.

Pages: by default a document is split into pages of about 90 seconds of audio; a page is closed
as soon as the audio of its items (end - beg of each clip) adds up to --seconds-per-page seconds
or more. With --seconds-per-page 0 a whole document is one page.

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

from pearmut_server_utils import generate_user_name

from instruction_html import instruction_html, esc
from nmee_protocol_v6_info import info as nmee_protocol_info

MIME = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg",
        ".m4a": "audio/mp4", ".opus": "audio/ogg"}

GRAY = "#555"


def note_block(label, text):
    """Italic, smaller block with a gray label, used for the gold transcript and the references."""
    return (f'<div style="font-style: italic; font-size: 0.85em;">'
            f'<span style="color: {GRAY};">{label}</span><br>'
            f'{esc(text)}</div>')


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
                              "severity": None, "category": None})
    return tgt, spans


def build_item(rec, args, refs, assets_url, stats):
    if not rec.get("audio"):
        print(f"WARNING: {rec.get('document')}: no \"audio\", skipped", file=sys.stderr)
        return None
    a = Path(rec["audio"])
    if not a.is_absolute():
        relative = Path(args.audio_relative_to or ".")
        clip = relative / Path(rec["audio"])
    else:
        clip = a
    rel = Path(*clip.parts[-2:]) if len(clip.parts) > 1 else Path(clip.name)
    if not clip.exists():
        stats["missing clips"] += 1
    elif args.copy_assets:
        dst = Path(args.copy_assets) / assets_url / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip, dst)

    src = (f'<audio controls src="{assets_url}/{rel.as_posix()}" '
           f'type="{MIME.get(clip.suffix.lower(), "audio/wav")}"></audio>')
    if rec.get("gold_transcript") and not args.no_gold:
        src += note_block("Gold transcript:", rec["gold_transcript"])
    for field, text in (find_reference(refs, rec).items() if refs else []):
        src += note_block(f"Reference ({esc(field)}):", text)

    tgt, spans = columns(rec, args)
    if not tgt:
        return None

    # only the fields Pearmut consumes; anything else would just bloat the logs
    item = {
        "item_id": f"{rec.get('document')}#{clip.stem.split('.')[-1]}",
        "src": src,
        "tgt": tgt,
        "instructions": instruction_html(rec, rec["targets"], tgt),
    }
    if spans and args.prefill:
        item["error_spans"] = dict(spans)
    stats["items"] += 1
    stats["spans"] += sum(len(v) for v in spans.values())
    return item


# ---------------------------------------------------------------- pages

def clip_seconds(rec):
    """Audio length of the clip of one record (end - beg); 0 if the timestamps are missing."""
    try:
        return max(0.0, float(rec["end"]) - float(rec["beg"]))
    except (KeyError, TypeError, ValueError):
        return 0.0


def build_pages(recs, args, refs, assets_url, stats):
    """Items of one document (sorted by start time) grouped into pages.

    With --seconds-per-page 0 the whole document is one page. Otherwise a page is closed once the
    audio of its items adds up to --seconds-per-page seconds or more."""
    pages, page, seconds = [], [], 0.0
    for rec in recs:
        item = build_item(rec, args, refs, assets_url, stats)
        if not item:
            continue
        page.append(item)
        seconds += clip_seconds(rec)
        if args.seconds_per_page and seconds >= args.seconds_per_page:
            pages.append(page)
            page, seconds = [], 0.0
    if page:
        pages.append(page)
    return pages

# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="JSONL of flagged harmful errors")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--audio-relative-to", help="directory the \"audio\" paths are relative to (default: current directory if \"audio\" is relative, absolute otherwise)")
    ap.add_argument("--campaign-id", default="harm_review")
    ap.add_argument("--assets-url", metavar="ASSETS_URL", 
                    help="URL prefix for clips (default: ./assets/<campaign-id>)")
    ap.add_argument("--copy-assets", metavar="ASSETS_DIR", #default="./data",
                    help="copy the needed clips to ASSETS_DIR/ASSETS_URL/")

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
    ap.add_argument("--prefill", action="store_true",
                    help="pre-highlight the flagged spans for debugging. Default: off")

    # campaign
    ap.add_argument("--users", type=int, default=1, help="number of annotator tasks (default: 1)")
    ap.add_argument("--usernames", type=str, nargs="+", default=[],
                    help="names of the annotators (default: auto-generated)")
    ap.add_argument("--partition", action="store_true",
                    help="split the documents across --users instead of giving each user all of them "
                         "(all pages of a document go to the same user)")
    ap.add_argument("--seconds-per-page", type=int, default=90,
                    help="number of audio seconds after which a page is wrapped, counted as the sum "
                         "of the clip lengths (end - beg) of the items on the page "
                         "(default: 90; 0 = the whole document per page)")
    ap.add_argument("--shuffle", choices=["keep", "on", "off"], default="off",
                    help="model shuffling: the columns are systems, so "
                         "shuffle them for fairer blind annotation. \"keep\" is what annotation protocol specifies. "
                         "(default: off)")
    ap.add_argument("--show-model-names", choices=["keep", "on", "off"], default="on",
                    help="label each column with its system name (default: on)")
    args = ap.parse_args()

    if args.seconds_per_page < 0:
        ap.error("--seconds-per-page must be >= 0 (0 = the whole document per page)")

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

    stats = defaultdict(int)
    doc_pages = []  # one entry per document: its list of pages (each page is a list of items)
    for doc in sorted(by_doc):
        pages = build_pages(sorted(by_doc[doc], key=lambda r: r.get("beg") or 0),
                            args, refs, assets_url, stats)
        if pages:
            doc_pages.append(pages)

    def flatten(docs):
        return [page for pages in docs for page in pages]

    if args.partition:
        tasks = [t for t in (flatten(doc_pages[i::args.users]) for i in range(args.users)) if t]
    else:
        tasks = [flatten(doc_pages) for _ in range(args.users)]
    n_pages = sum(len(p) for p in doc_pages)

    info = nmee_protocol_info
    if args.shuffle != "keep":
        info["shuffle"] = args.shuffle == "on"
    if args.show_model_names != "keep":
        info["show_model_names"] = args.show_model_names == "on"
    if args.usernames:
        usernames = args.usernames
        while len(usernames) < args.users:
            usernames.append(generate_user_name(usernames))
        info["users"] = usernames
    if args.prefill:
        info["instructions"] = (info.get("instructions", "")
            + '\n\n<p style="color: red;"><b>The spans are pre-filled because this is view-only '
              'for debugging. Do not annotate.</b></p>')

    json.dump({"info": info, "campaign_id": args.campaign_id, "data": tasks},
              open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if args.copy_assets:
        print(f"clips copied to {Path(args.copy_assets).resolve() / assets_url}", file=sys.stderr)
        print(f"clips referenced as {assets_url}/<document>/<clip>", file=sys.stderr)
    if stats["missing clips"]:
        print(f"WARNING: {stats['missing clips']} clip(s) not found (--audio-relative-to {args.audio_relative_to})",
              file=sys.stderr)
    print(f"{len(doc_pages)} documents, {n_pages} pages, {stats['items']} items, {n_errors} flagged span(s), "
          f"{stats['spans']} pre-filled, {len(tasks)} task(s) "
          f"{'partitioned' if args.partition else 'replicated'} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()