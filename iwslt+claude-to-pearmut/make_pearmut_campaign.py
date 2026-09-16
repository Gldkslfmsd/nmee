#!/usr/bin/env python3
"""Build a Pearmut campaign from annotations.jsonl, using the custom NMEE protocol
(ESA, no sliders, no severities, word-level spans, harm-annotation instructions), i.e. the "info"
block of custom_nmee_demo.json. Use --template to take the "info" block from another campaign file.

- One Pearmut document (= one page) per audio file, containing only the annotated segments.
- Each item: source = audio clip (+ optionally the overlapping gold transcript / reference translation),
  target = the translated segment, `instructions` = the suggested error(s): span, intended meaning,
  harm type, confidence (harmful / borderline), likely error source (ASR / MT) and explanation.
- Suggested spans are pre-filled (severity --prefill-severity, default "major" = red); --no-prefill disables it.
- --confidence high|low|all selects harmful (high) or borderline (low) suggestions, or both.

annotations.jsonl fields used: filename, segment (0-based index into <doc>.en.jsonl), segment_text,
span, span_start, span_end, intended, harm_types, confidence, error_source, explanation, asr_context.

Usage:
    python make_pearmut_campaign.py annotations.jsonl \
        --translations-dir outputs/iwslt26-cs-dev --segments-dir segments \
        --ref-yaml iwslt26-cs-dev.yaml --ref-cs iwslt26-cs-dev.cs --ref-en iwslt26-cs-dev.en \
        --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json
    pearmut add -o campaign.json

Clips (from segment_audio.py) are expected at SEGMENTS_DIR/<doc>/<doc>.<NNNN>.wav and referenced as
./assets/<campaign-id>/<doc>/<doc>.<NNNN>.wav (change with --assets-url).
"""
import argparse
import html
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# "info" block of custom_nmee_demo.json
DEFAULT_INFO = {
    "assignment": "task-based",
    "protocol": "ESA",
    "sliders": [],
    "mqm_severities": [],
    "instructions": "<p><b>Brief summary:</b> Highlight comprehension (adequacy) errors that are also <b>harmful</b>, i.e. they would cause a problem beyond the misunderstanding itself: someone would have to correct it or apologise for it, be offended or embarrassed by it, laugh at it, or be at risk if they acted on it.</p> <details> <summary><b>Full instructions</b> (click to expand)</summary> <p>Assume a live communication event with a speaker and an audience. They are in the same room (so it is <b>not</b> an online meeting), the translation is displayed on a large screen in the room, and the audience reads it.</p> <p>Make sure you understand the specifics of the domain and of the particular situation of the communication event.<br> <i>Example:</i> The conference “Média a Ukrajina” (Media and Ukraine) took place in Prague on June 22, 2023. The main topic was the role of public-service media in the war in Ukraine. Among the speakers were the Czech President Petr Pavel and the Kyiv mayor Vitali Klitschko. The programme, production and moderation were adapted to live broadcasting on Czech Radio. Some speeches were pre-recorded.</p> <p>Below are audio segments and translations. Your task is to <b>highlight spans that contain a harmful comprehension error</b>.</p> <h3>Comprehension errors</h3><p>A comprehension error changes or obscures the meaning, or misleads the viewer. Examples:</p> <ul> <li>intended “I am an assistant professor” → output “I am a professional assistant.”</li> <li>intended “a public media service” → output “pubic media service”: a typo resulting in an unintended word.</li> <li>intended “thank you” → output “thank you thank you thank you thank you thank you thank you thank you”: repeated so many times that it would be considered awkward.</li> <li>Czech: “Žanno Němcovová, řekl byste”: addressing a woman with a masculine verb form.</li> <li>Czech, wrong register: “díky ti, pane prezidente”: informal instead of formal.</li> </ul><p><i>Negative examples:</i> grammar or fluency errors such as a missing space, wrong capitalisation, punctuation errors, wrong spelling, wrong case, typos, etc. are <b>not</b> comprehension errors, unless they change the literal meaning.</p> <h3>Harmful errors</h3><p>We consider an error harmful if it would cause a problem beyond the misunderstanding itself: someone would have to correct it or apologise for it, be offended or embarrassed by it, laugh at it, or be at risk if they acted on it.</p> <p>We presume the following types of harm. One error can belong to multiple types, further specification is not needed.</p> <p><b>False attribution:</b> the translation gives a person the wrong identity, gender, role, relationship, or characteristics. Examples:</p> <ul> <li>intended “I'd like to thank Andrea for her introductory speech” → output “…for <span style=\"background: #ffa6a6\">his</span> introductory speech.” Andrea is a woman.</li> <li>intended “chairwoman of a committee” → output “<span style=\"background: #ffa6a6\">daughter of a priest</span>”.</li> <li>a claim about the poor state of prisons “in <span style=\"background: #ffa6a6\">British</span> states” instead of “in Baltic states”.</li> </ul><p><b>Offensive:</b> a proper name is mishandled, a name is declined into the wrong gender, or the output is disrespectful to a person or group. Examples:</p> <ul> <li>intended “Aisha Khan” → output “<span style=\"background: #ffa6a6\">asylum con</span>”. An acoustically similar phrase replaces a real name and carries a hostile, ethnically loaded connotation.</li> <li>intended “Ukrainian civilians were killed” → output “<span style=\"background: #ffa6a6\">Russian</span> civilians were killed”. The victims are misidentified, in front of an audience that may include Ukrainians.</li> </ul><p><b>Embarrassing / laughable:</b> the error introduces explicit, absurd or unintentionally funny content. Examples:</p> <ul> <li>intended “…the meeting is scheduled for 4 p.m., ish…” → output “�
    "word_level": True,
    "show_alignment": False
}


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


def instruction_html(anns, show_asr):
    parts = []
    for a in anns:
        p = (f"<b>Suggested ({a['confidence']}):</b> “{esc(a['span'])}” → intended: “{esc(a['intended'])}”. "
             f"<i>{esc(', '.join(a['harm_types']))}; likely source: {esc(a['error_source'])}.</i> {esc(a['explanation'])}")
        if show_asr and a.get("asr_context"):
            p += f"<br><small>ASR: {esc(a['asr_context'])}</small>"
        parts.append(p)
    return "<br>".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations")
    ap.add_argument("--translations-dir", required=True)
    ap.add_argument("--suffix", default=".en.jsonl")
    ap.add_argument("--segments-dir", required=True, help="output dir of segment_audio.py")
    ap.add_argument("--campaign-id", default="iwslt26_csen_harm_review")
    ap.add_argument("--assets-url", help="URL prefix for clips (default: ./assets/<campaign-id>)")
    ap.add_argument("--copy-assets", metavar="ASSETS_DIR", help="copy needed clips to ASSETS_DIR/<campaign-id>/")
    ap.add_argument("--ref-yaml", help="reference segmentation with offsets (iwslt26-cs-dev.yaml)")
    ap.add_argument("--ref-cs", help="gold transcript aligned with --ref-yaml")
    ap.add_argument("--ref-en", help="reference translation aligned with --ref-yaml")
    ap.add_argument("--min-overlap", type=float, default=0.1)
    ap.add_argument("--show-asr", action="store_true", help="show the ASR line used as evidence")
    ap.add_argument("--confidence", choices=["all", "high", "low"], default="all",
                    help="high = only suggestions marked harmful, low = only borderline ones (default: all)")
    ap.add_argument("--template", help="campaign JSON whose \"info\" block is copied verbatim "
                    "(e.g. custom_nmee_demo.json; default: the built-in copy of that block)")
    ap.add_argument("--no-prefill", action="store_true", help="don't pre-highlight suggested spans")
    ap.add_argument("--prefill-severity", default="major",
                    help="severity stored in pre-filled spans (the protocol has no severity buttons; default: major)")
    ap.add_argument("--model-name", default="system")
    ap.add_argument("--users", type=int, default=1)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    assets_url = (args.assets_url or f"./assets/{args.campaign_id}").rstrip("/")
    refs = load_refs(args.ref_yaml, args.ref_cs, args.ref_en) if args.ref_yaml and args.ref_cs else None

    by_doc = defaultdict(lambda: defaultdict(list))
    for l in open(args.annotations, encoding="utf-8"):
        if not l.strip():
            continue
        a = json.loads(l)
        wanted = {"all": None, "high": "harmful", "low": "borderline"}[args.confidence]
        if wanted and a["confidence"] != wanted:
            continue
        by_doc[Path(a["filename"]).stem][a["segment"]].append(a)

    documents, n_items = [], 0
    for stem in sorted(by_doc):
        segs = [json.loads(l) for l in open(Path(args.translations_dir) / f"{stem}{args.suffix}", encoding="utf-8") if l.strip()]
        doc = []
        for k in sorted(by_doc[stem]):
            anns = by_doc[stem][k]
            # sanity check: the annotated text must be the k-th segment; otherwise look it up by text
            if k >= len(segs) or " ".join(segs[k]["segment"].split()) != " ".join(anns[0]["segment_text"].split()):
                hits = [i for i, s in enumerate(segs) if " ".join(s["segment"].split()) == " ".join(anns[0]["segment_text"].split())]
                if len(hits) != 1:
                    print(f"WARNING: {stem}#{k}: segment text not found in jsonl, skipped", file=sys.stderr)
                    continue
                print(f"NOTE: {stem}: segment {k} is jsonl line {hits[0]}", file=sys.stderr)
                k = hits[0]
            seg, text = segs[k], segs[k]["segment"]

            clip_name = f"{stem}.{k:04d}.wav"
            clip = Path(args.segments_dir) / stem / clip_name
            if not clip.exists():
                print(f"WARNING: missing clip {clip}", file=sys.stderr)
            elif args.copy_assets:
                dst = Path(args.copy_assets) / args.campaign_id / stem
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, dst / clip_name)

            src = f'<audio controls src="{assets_url}/{stem}/{clip_name}" type="audio/wav"></audio>'
            ref_cs = ref_en = None
            if refs is not None and stem in refs and seg.get("start") is not None:
                ov = overlapping(refs[stem], seg["start"], seg["end"], args.min_overlap)
                ref_cs = " ".join(r[2] for r in ov)
                ref_en = " ".join(r[3] for r in ov if r[3]) or None
                src += f"<br><i>Gold transcript (may cover more than this clip):</i><br>{esc(ref_cs)}"
                if ref_en:
                    src += f"<br><i>Reference translation:</i><br>{esc(ref_en)}"

            spans = []
            for a in anns:
                s0 = a["span_start"] if text[a["span_start"]:a["span_end"]] == a["span"] else text.find(a["span"])
                if s0 < 0:
                    print(f"WARNING: span {a['span']!r} not in {stem}#{k}", file=sys.stderr)
                    continue
                spans.append({"start_i": s0, "end_i": s0 + len(a["span"]) - 1,  # inclusive end
                              "severity": args.prefill_severity, "category": None})

            item = {
                "item_id": f"{stem}#{k:04d}",
                "src": src,
                "tgt": {args.model_name: text},
                "instructions": instruction_html(anns, args.show_asr),
                "doc_id": stem, "segment": k, "start": seg.get("start"), "end": seg.get("end"),
                "ref_cs": ref_cs, "ref_en": ref_en, "suggestions": anns,
            }
            if spans and not args.no_prefill:
                item["error_spans"] = {args.model_name: spans}
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
    print(f"clips referenced as {assets_url}/<doc>/<doc>.<NNNN>.wav", file=sys.stderr)
    print(f"{len(documents)} documents, {n_items} items, {args.users} task(s) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
