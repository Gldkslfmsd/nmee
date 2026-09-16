#!/usr/bin/env python3
"""Build a Pearmut campaign (MQM protocol with a custom harm taxonomy) from resolved annotations.

- One Pearmut document (= one page) per audio file; only the annotated segments are included.
- Each item: source = audio clip of the segment (+ the overlapping gold transcript if the annotations
  were resolved with --ref-yaml/--ref-text), target = the translated segment,
  `instructions` = the suggested error(s) with explanation, shown above the segment.
- Suggested spans are pre-filled as error spans (harmful -> "major", borderline -> "minor"),
  with the category "Harmful/<type>". Use --no-prefill to show only the explanation.

Usage:
    python make_pearmut_campaign.py annotations.jsonl \
        --translations-dir ~/work/uedin/mtm26/nmee/outputs/iwslt26-cs-dev \
        --segments-dir segments \
        --copy-assets "${PEARMUT_ROOT:-.}/data/assets" \
        --campaign-id iwslt26_csen_harm_review \
        -o campaign.json
    pearmut add campaign.json && pearmut run

Clips are expected at SEGMENTS_DIR/<doc>/<doc>.<NNNN>.wav (output of segment_audio.py) and are
referenced as ASSETS_URL/<doc>/<doc>.<NNNN>.wav (default ASSETS_URL: /assets/<campaign-id>).
--copy-assets copies the needed clips to ASSETS_DIR/<campaign-id>/<doc>/.
"""
import argparse
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HARM_TYPES = [
    "False attribution",
    "Offensive",
    "Embarrassing or laughable",
    "Derailing or contresens",
    "Safety, health or legal risk",
    "Other",
]

GLOBAL_INSTRUCTIONS = """
<b>Harmful errors in speech translation (review of suggested annotations)</b><br>
Imagine a live event in the Czech Chamber of Deputies; the English translation is shown on a large screen.
Listen to the segment and highlight <b>comprehension errors that are also harmful</b>: someone would have to
correct or apologise for them, be offended or embarrassed, laugh at them, or be at risk if they acted on them.<br>
Each segment shows a <i>suggested</i> error with an explanation; it is pre-highlighted.
<b>Major</b> = suggested as harmful, <b>Minor</b> = borderline. Keep, change, or delete the suggestion,
and add anything that was missed. Grammar/fluency problems that don't change meaning are not errors.<br>
Categories: <b>Harmful/&lt;type&gt;</b> for harmful errors; <b>Not harmful/Comprehension error</b> if the error
changes meaning but causes no harm beyond the misunderstanding.
""".strip()


def esc(s):
    return html.escape(s, quote=False)


def instruction_html(anns):
    parts = []
    for a in anns:
        label = "borderline" if a["borderline"] else "harmful"
        parts.append(
            f"<b>Suggested ({label}):</b> “{esc(a['span'])}” → intended: “{esc(a['intended'])}”. "
            f"<i>{esc(', '.join(a['types']))}.</i> {esc(a['explanation'])}"
        )
    return "<br>".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="resolved annotations.jsonl (with filename and segment)")
    ap.add_argument("--translations-dir", required=True)
    ap.add_argument("--suffix", default=".en.jsonl")
    ap.add_argument("--segments-dir", required=True, help="output dir of segment_audio.py")
    ap.add_argument("--campaign-id", default="iwslt26_csen_harm_review")
    ap.add_argument("--assets-url", help="URL prefix for clips (default: /assets/<campaign-id>)")
    ap.add_argument("--copy-assets", metavar="ASSETS_DIR",
                    help="Pearmut data/assets dir; clips are copied to ASSETS_DIR/<campaign-id>/")
    ap.add_argument("--include", choices=["all", "harmful", "borderline"], default="all")
    ap.add_argument("--no-prefill", action="store_true", help="don't pre-fill error spans")
    ap.add_argument("--no-ref", action="store_true", help="don't show the gold Czech transcript")
    ap.add_argument("--model-name", default="system")
    ap.add_argument("--users", type=int, default=1, help="number of identical tasks (annotator URLs)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    assets_url = (args.assets_url or f"/assets/{args.campaign_id}").rstrip("/")
    seg_dir = Path(args.segments_dir)

    # group annotations by document and segment
    by_doc = defaultdict(lambda: defaultdict(list))
    for l in open(args.annotations, encoding="utf-8"):
        if not l.strip():
            continue
        a = json.loads(l)
        if a.get("filename") is None:
            print(f"skipping unresolved annotation: {a['span']!r}", file=sys.stderr)
            continue
        if args.include == "harmful" and a["borderline"]:
            continue
        if args.include == "borderline" and not a["borderline"]:
            continue
        by_doc[Path(a["filename"]).stem][a["segment"]].append(a)

    documents = []
    n_items = 0
    for stem in sorted(by_doc):
        jf = Path(args.translations_dir) / f"{stem}{args.suffix}"
        segs = [json.loads(l) for l in open(jf, encoding="utf-8") if l.strip()]
        doc = []
        for k in sorted(by_doc[stem]):
            anns = by_doc[stem][k]
            seg = segs[k]
            text = seg["segment"]
            clip_name = f"{stem}.{k:04d}.wav"
            clip = seg_dir / stem / clip_name
            if not clip.exists():
                print(f"WARNING: missing clip {clip} (run segment_audio.py)", file=sys.stderr)
            elif args.copy_assets:
                dst = Path(args.copy_assets) / args.campaign_id / stem
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, dst / clip_name)

            spans = []
            for a in anns:
                if text[a["span_start"]:a["span_end"]] != a["span"]:
                    # the jsonl text may differ in whitespace from the concat file: re-locate
                    pos = text.find(a["span"])
                    if pos < 0:
                        print(f"WARNING: span {a['span']!r} not in {stem}#{k}", file=sys.stderr)
                        continue
                    a["span_start"], a["span_end"] = pos, pos + len(a["span"])
                spans.append({
                    "start_i": a["span_start"],
                    "end_i": a["span_end"] - 1,  # Pearmut end index is inclusive
                    "severity": "minor" if a["borderline"] else "major",
                    "category": f"Harmful/{a['types'][0]}",
                })

            src = f'<audio controls preload="none" src="{assets_url}/{stem}/{clip_name}" type="audio/wav"></audio>'
            ref_text = anns[0].get("ref_text")
            if ref_text and not args.no_ref:
                src += f"<br><i>Czech transcript (reference segment, may cover more than this clip):</i><br>{esc(ref_text)}"
            item = {
                "item_id": f"{stem}#{k:04d}",
                "src": src,
                "tgt": {args.model_name: text},
                "instructions": instruction_html(anns),
                # extra keys are stored in the logs
                "doc_id": stem,
                "segment": k,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "suggestions": anns,
            }
            if not args.no_prefill and spans:
                item["error_spans"] = {args.model_name: spans}
            doc.append(item)
            n_items += 1
        if doc:
            documents.append(doc)

    campaign = {
        "info": {
            "assignment": "task-based",
            "protocol": "MQM",
            "mqm_categories": {
                "Harmful": HARM_TYPES,
                "Not harmful": ["Comprehension error"],
            },
            "mqm_severities": ["Minor", "Major"],
            "instructions": GLOBAL_INSTRUCTIONS,
            "shuffle": False,
            "users": args.users,
        },
        "campaign_id": args.campaign_id,
        "data": [documents for _ in range(args.users)],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(campaign, f, ensure_ascii=False, indent=2)
    print(f"{len(documents)} documents, {n_items} items, {args.users} task(s) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
