#!/usr/bin/env python3
"""Build a Pearmut campaign from annotations.jsonl, using the custom NMEE protocol
(ESA, no sliders, no severities, word-level spans, harm-annotation instructions), i.e. the "info"
block of custom_nmee_demo.json. Use --template to take the "info" block from another campaign file.

- One Pearmut document (= one page) per doc_id, containing only the annotated segments.
- Each item: source = audio clip + gold transcript (+ reference translation), target = one text per
  system from "targets", `instructions` = the suggested error(s) per target: span, intended meaning,
  harm types, confidence, likely error source and explanation.
- Suggested spans are pre-filled (severity --prefill-severity, default "major" = red); --no-prefill disables it.
- --confidence high|low|all keeps only targets marked harmful (high) or borderline (low), or both.

annotations.jsonl: one line per segment, in the agreed format:
  doc_id, filename, segment_filename and/or segment and/or orig_start/orig_end,
  asr, gold_transcript,
  targets: [{tgt_lan, system, text, reference, span, span_start, span_end, intended, harm_types,
             confidence, borderline, explanation, error_source, ...}]

Usage:
    python make_pearmut_campaign.py annotations.jsonl --segments-dir segments \
        --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json
    pearmut add -o campaign.json

Optional: --translations-dir outputs/iwslt26-cs-dev checks that each target text is the given segment of
<doc>.en.jsonl and takes its timestamps; with --ref-yaml/--ref-cs/--ref-en the gold transcript and
reference are then re-aligned by time instead of using the values stored in annotations.jsonl.

Clips (from segment_audio.py) are taken from SEGMENTS_DIR/<segment_filename> (default
./<doc>/<doc>.<NNNN>.wav) and referenced as ./assets/<campaign-id>/<doc>/<doc>.<NNNN>.wav (see --assets-url).
"""
import argparse
import html
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

with open("../custom_nmee_demo.json","r") as f:
    DEFAULT_INFO = json.load(f)["info"]

def esc(s):
    return html.escape(s, quote=False)


def load_refs(yaml_path, cs_path, en_path):
    """{doc_stem: [(start, end, cs, en), ...]} from the reference segmentation."""
    cs = open(cs_path, encoding="utf-8").read().rstrip("\n").split("\n")
    en = open(en_path, encoding="utf-8").read().rstrip("\n").split("\n") if en_path else [None] * len(cs)
    try:
        import yaml
        entries = yaml.safe_load(open(yaml_path, encoding="utf-8"))
    except ImportError:  # one flow mapping per line
        entries = []
        for l in open(yaml_path, encoding="utf-8"):
            if l.strip():
                d = dict(re.findall(r"(\w+):\s*([^,}]+)", l))
                entries.append({"duration": float(d["duration"]), "offset": float(d["offset"]), "wav": d["wav"].strip()})
    if not (len(entries) == len(cs) == len(en)):
        sys.exit(f"reference files differ in length: yaml {len(entries)}, cs {len(cs)}, en {len(en)}")
    refs = defaultdict(list)
    for e, c, t in zip(entries, cs, en):
        refs[Path(e["wav"]).stem].append((e["offset"], e["offset"] + e["duration"], c, t))
    return refs


def overlapping(refs, start, end, min_overlap):
    out = [r for r in refs if min(end, r[1]) - max(start, r[0]) > min_overlap]
    return out or [min(refs, key=lambda r: abs((r[0] + r[1]) / 2 - (start + end) / 2))]


def instruction_html(rec, targets, show_asr, multi_system):
    parts = []
    for t in targets:
        who = f"[{esc(t['system'])}] " if multi_system else ""
        p = (f"{who}<b>Suggested ({esc(t.get('confidence', 'unknown'))}):</b> “{esc(t['span'])}” → "
             f"intended: “{esc(t.get('intended', ''))}”. <i>{esc(', '.join(t.get('harm_types', [])))}; "
             f"likely source: {esc(t.get('error_source', 'unknown'))}.</i> {esc(t.get('explanation', ''))}")
        parts.append(p)
    if show_asr and rec.get("asr"):
        parts.append(f"<small>ASR: {esc(rec['asr'])}</small>")
    return "<br>".join(parts)


def norm(s):
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations")
    ap.add_argument("--segments-dir", required=True, help="output dir of segment_audio.py")
    ap.add_argument("--translations-dir", help="dir with <doc>.en.jsonl (optional: sanity check + timestamps)")
    ap.add_argument("--suffix", default=".en.jsonl")
    ap.add_argument("--campaign-id", default="iwslt26_csen_harm_review")
    ap.add_argument("--assets-url", help="URL prefix for clips (default: ./assets/<campaign-id>)")
    ap.add_argument("--copy-assets", metavar="ASSETS_DIR", help="copy needed clips to ASSETS_DIR/<campaign-id>/")
    ap.add_argument("--ref-yaml", help="re-align gold/reference by time (needs timestamps)")
    ap.add_argument("--ref-cs", help="gold transcript aligned with --ref-yaml")
    ap.add_argument("--ref-en", help="reference translation aligned with --ref-yaml")
    ap.add_argument("--min-overlap", type=float, default=0.1)
    ap.add_argument("--no-gold", action="store_true", help="don't show the gold transcript")
    ap.add_argument("--no-reference", action="store_true", help="don't show the reference translation")
    ap.add_argument("--show-asr", action="store_true", help="show the ASR text of the segment")
    ap.add_argument("--confidence", choices=["all", "high", "low"], default="all",
                    help="high = only targets marked harmful, low = only borderline ones (default: all)")
    ap.add_argument("--template", help="campaign JSON whose \"info\" block is copied verbatim "
                    "(e.g. custom_nmee_demo.json; default: the built-in copy of that block)")
    ap.add_argument("--no-prefill", action="store_true", help="don't pre-highlight suggested spans")
    ap.add_argument("--prefill-severity", default="major",
                    help="severity stored in pre-filled spans (the protocol has no severity buttons; default: major)")
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    assets_url = (args.assets_url or f"./assets/{args.campaign_id}").rstrip("/")
    refs = load_refs(args.ref_yaml, args.ref_cs, args.ref_en) if args.ref_yaml and args.ref_cs else None
    wanted = {"all": None, "high": "harmful", "low": "borderline"}[args.confidence]

    by_doc = defaultdict(list)
    for l in open(args.annotations, "r"):
        if not l.strip():
            continue
        rec = json.loads(l)
        rec["targets"] = [t for t in rec.get("targets", []) if not wanted or t.get("confidence") == wanted]
        if rec["targets"]:
            by_doc[rec["doc_id"]].append(rec)

    jsonl_cache = {}
    documents, n_items = [], 0
    for doc_id in sorted(by_doc):
        doc = []
        recs = sorted(by_doc[doc_id], key=lambda r: (r.get("segment", -1), r.get("orig_start") or 0))
        for rec in recs:
            stem = Path(rec["filename"]).stem
            k = rec.get("segment")
            start, end = rec.get("orig_start"), rec.get("orig_end")

            # optional check against the system output + timestamps
            if args.translations_dir and k is not None:
                if stem not in jsonl_cache:
                    p = Path(args.translations_dir) / f"{stem}{args.suffix}"
                    jsonl_cache[stem] = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if p.exists() else None
                segs = jsonl_cache[stem]
                text0 = norm(rec["targets"][0]["text"])
                if segs is not None:
                    if k >= len(segs) or norm(segs[k]["segment"]) != text0:
                        print(f"WARNING: {doc_id}#{k}: target text differs from {stem}{args.suffix} line {k}", file=sys.stderr)
                    elif start is None:
                        start, end = segs[k].get("start"), segs[k].get("end")

            # clip
            seg_file = rec.get("segment_filename") or (f"./{stem}/{stem}.{k:04d}.wav" if k is not None else None)
            if seg_file is None:
                print(f"WARNING: {doc_id}: no segment_filename/segment, skipped", file=sys.stderr)
                continue
            rel = Path(seg_file)
            clip = Path(args.segments_dir) / rel
            if not clip.exists():
                print(f"WARNING: missing clip {clip}", file=sys.stderr)
            elif args.copy_assets:
                dst = Path(args.copy_assets) / args.campaign_id / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, dst)
            url = f"{assets_url}/{rel.as_posix().removeprefix('./')}"

            # gold transcript / reference: stored values, or re-aligned by time
            gold = rec.get("gold_transcript")
            references = {}
            for t in rec["targets"]:
                if t.get("reference"):
                    references.setdefault((t.get("tgt_lan"), t["reference"]), None)
            if refs is not None and stem in refs and start is not None:
                ov = overlapping(refs[stem], start, end, args.min_overlap)
                gold = " ".join(r[2] for r in ov)
                ref_en = " ".join(r[3] for r in ov if r[3])
                references = {("en", ref_en): None} if ref_en else {}

            src = f'<audio controls src="{url}" type="audio/wav"></audio>'
            if gold and not args.no_gold:
                src += f"<br><i>Gold transcript (may cover more than this clip):</i><br>{esc(gold)}"
            if references and not args.no_reference:
                for lan, r in references:
                    src += f"<br><i>Reference translation{f' ({esc(lan)})' if lan else ''}:</i><br>{esc(r)}"

            # targets: one text per system, spans per system
            tgt, spans = {}, defaultdict(list)
            for t in rec["targets"]:
                system, text = t["system"], t["text"]
                if tgt.setdefault(system, text) != text:
                    print(f"WARNING: {doc_id}#{k}: system {system} has different texts, keeping the first", file=sys.stderr)
                    continue
                s0 = t["span_start"] if text[t["span_start"]:t["span_end"]] == t["span"] else text.find(t["span"])
                if s0 < 0:
                    print(f"WARNING: span {t['span']!r} not in {doc_id}#{k} ({system})", file=sys.stderr)
                    continue
                spans[system].append({"start_i": s0, "end_i": s0 + len(t["span"]) - 1,  # inclusive end
                                      "severity": args.prefill_severity, "category": None})

            item = {
                "item_id": f"{doc_id}#{k:04d}" if k is not None else f"{doc_id}@{start}",
                "src": src,
                "tgt": tgt,
                "instructions": instruction_html(rec, rec["targets"], args.show_asr, len(tgt) > 1),
                # extra keys are kept in the logs
                "doc_id": doc_id, "filename": rec["filename"], "segment": k,
                "segment_filename": seg_file, "orig_start": start, "orig_end": end,
                "asr": rec.get("asr"), "gold_transcript": gold, "targets": rec["targets"],
            }
            if spans and not args.no_prefill:
                item["error_spans"] = dict(spans)
            doc.append(item)
            n_items += 1
        if doc:
            documents.append(doc)

    # the "info" block is used verbatim, nothing is added or changed
    info = json.load(open(args.template, encoding="utf-8"))["info"] if args.template else DEFAULT_INFO
    campaign = {
        "info": info,
        "campaign_id": args.campaign_id,
        "data": [documents for _ in range(args.users)],
    }
    json.dump(campaign, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if args.copy_assets:
        print(f"clips copied to {Path(args.copy_assets).resolve() / args.campaign_id}", file=sys.stderr)
    print(f"clips referenced as {assets_url}/<segment_filename>", file=sys.stderr)
    print(f"{len(documents)} documents, {n_items} items, {args.users} task(s) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
