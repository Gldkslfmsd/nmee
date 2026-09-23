#!/usr/bin/env python3
"""Build a Pearmut campaign from the Earnings25 English-ASR harm annotations.

Adapted from iwslt+claude-to-pearmut/make_pearmut_campaign.py, which is overspecific to Dávid's
IWSLT26 dev set (yaml segmentation, .cs/.en reference files, wav clips). Earnings25 has none of
those: the gold transcript is already stored per segment in the annotations, there is no MT stage
and so no reference translation, and the clips are mp3.

Differences from the IWSLT version, beyond dropping the yaml/reference re-alignment:
  - clips are mp3 and are located through `segment_filename` (keyed by clip_id, not recording id);
  - pre-filled spans carry no severity: the NMEE protocol has no severity buttons, so a severity
    would only paint a colour prior onto the annotator's screen;
  - the gold transcript is shown next to the audio (--no-gold turns it off);
  - the "info" template is resolved relative to this file, not to the working directory;
  - --severity / --confidence select a stratum, and --partition splits the documents across
    annotators instead of giving everyone the same ones.

This is ASR-only: `tgt` is the English Canary transcript itself, and every error_source is "ASR".

Usage:
    python make_pearmut_campaign.py annotations.jsonl --clips-dir clips \
        --severity high,medium --copy-assets "${PEARMUT_ROOT:-.}/data/assets" -o campaign.json
    pearmut add -o campaign.json
"""
import argparse
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "custom_nmee_demo.json"


def esc(s):
    return html.escape(s, quote=False)


def instruction_html(rec, targets, show_asr, multi_system):
    parts = []
    for t in targets:
        who = f"[{esc(t['system'])}] " if multi_system else ""
        if t.get("source") == "dspy-divergencies":
            # a vote of three LLMs over five languages, with its own error taxonomy; harm_types
            # is always ["Other"] there, so printing it would be noise
            n = t.get("num_models_reporting", 1)
            models = ", ".join(t.get("models_reporting", [])) or "?"
            if t.get("span_whole_output"):
                what = "<b>whole output flagged</b> (no span pinpointed)"
            else:
                what = f"“{esc(t['span'])}”"
            cls = f" {esc(t['error_class'])};" if t.get("error_class") else ""
            parts.append(
                f"{who}<b>Divergence ({n} model{'s' if n != 1 else ''}):</b> {what}."
                f"<i>{cls} {esc(models)}.</i> {esc(t.get('explanation', ''))}"
            )
            continue
        if t["span"]:
            what = f"“{esc(t['span'])}” → intended: “{esc(t.get('intended', ''))}”"
        else:  # pure deletion: nothing to quote, the words are simply absent
            what = f"missing: “{esc(t.get('intended', ''))}”"
        parts.append(
            f"{who}<b>Suggested ({esc(t.get('confidence', 'unknown'))}):</b> {what}. "
            f"<i>{esc(', '.join(t.get('harm_types', [])))}; "
            f"likely source: {esc(t.get('error_source', 'unknown'))}.</i> {esc(t.get('explanation', ''))}"
        )
    if show_asr and rec.get("asr"):
        parts.append(f"<small>ASR: {esc(rec['asr'])}</small>")
    return "<br>".join(parts)


def select(targets, confidence, severities, langs, sources, min_models):
    out = targets
    if confidence != "all":
        out = [t for t in out if t.get("confidence") == confidence]
    if severities:
        # only the asr-harm targets carry a severity; targets without one are not filtered out
        out = [t for t in out if "severity" not in t or t["severity"] in severities]
    if langs:
        out = [t for t in out if t.get("tgt_lan") in langs]
    if sources:
        out = [t for t in out if t.get("source", "asr-harm") in sources]
    if min_models > 1:
        out = [t for t in out if t.get("num_models_reporting", min_models) >= min_models]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", help="annotations.jsonl from normalize_annotations.py")
    ap.add_argument("--clips-dir", required=True, help="output dir of cut_clips.py")
    ap.add_argument("--campaign-id", default="earnings25_en_asr_harm_review")
    ap.add_argument("--assets-url", help="URL prefix for clips (default: ./assets/<campaign-id>)")
    ap.add_argument("--copy-assets", metavar="ASSETS_DIR", help="copy needed clips to ASSETS_DIR/<campaign-id>/")
    ap.add_argument("--confidence", choices=["all", "harmful", "borderline"], default="all")
    ap.add_argument("--severity", help="comma-separated subset of high,medium,low (default: all); "
                    "only applies to targets that carry a severity, i.e. the asr-harm ones")
    ap.add_argument("--lang", help="comma-separated target languages to keep, e.g. cs or en,de")
    ap.add_argument("--source", help="comma-separated subset of asr-harm,dspy-divergencies")
    ap.add_argument("--min-models", type=int, default=1,
                    help="keep divergence targets reported by at least this many models (default: 1)")
    ap.add_argument("--shuffle", choices=["keep", "on", "off"], default="keep",
                    help="override the template's model shuffling; columns are languages in a "
                    "merged campaign, so shuffling them only confuses the annotator (default: keep)")
    ap.add_argument("--show-model-names", choices=["keep", "on", "off"], default="keep",
                    help="label each output column; needed when the columns are languages")
    ap.add_argument("--no-gold", action="store_true", help="don't show the gold transcript")
    ap.add_argument("--show-asr", action="store_true", help="repeat the ASR text in the instructions")
    ap.add_argument("--no-prefill", action="store_true", help="don't pre-highlight suggested spans")
    ap.add_argument("--slim", action="store_true",
                    help="omit the echoed targets/asr/gold_transcript from each item. Pearmut keeps "
                    "any extra key in the annotation logs, which is handy but roughly quadruples "
                    "the campaign file; item_id joins back to the annotations file anyway.")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                    help=f"campaign JSON whose \"info\" block is copied verbatim (default: {DEFAULT_TEMPLATE})")
    ap.add_argument("--users", type=int, default=1, help="number of annotator tasks (default: 1)")
    ap.add_argument("--partition", action="store_true",
                    help="split the documents across --users instead of giving each user all of them")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    assets_url = (args.assets_url or f"./assets/{args.campaign_id}").rstrip("/")
    severities = set(args.severity.split(",")) if args.severity else None
    langs = set(args.lang.split(",")) if args.lang else None
    sources = set(args.source.split(",")) if args.source else None
    if severities and not severities <= {"high", "medium", "low"}:
        sys.exit(f"--severity must be a subset of high,medium,low (got {sorted(severities)})")

    by_doc = defaultdict(list)
    n_errors = 0
    for line in open(args.annotations, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["targets"] = select(rec.get("targets", []), args.confidence, severities,
                                langs, sources, args.min_models)
        if rec["targets"]:
            by_doc[rec["doc_id"]].append(rec)
            n_errors += len(rec["targets"])

    documents, n_items, n_spans, missing = [], 0, 0, 0
    for doc_id in sorted(by_doc):
        doc = []
        for rec in sorted(by_doc[doc_id], key=lambda r: r["segment"]):
            rel = Path(rec["segment_filename"])
            clip = Path(args.clips_dir) / rel
            if not clip.exists():
                print(f"WARNING: missing clip {clip}", file=sys.stderr)
                missing += 1
            elif args.copy_assets:
                dst = Path(args.copy_assets) / args.campaign_id / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, dst)
            url = f"{assets_url}/{rel.as_posix().removeprefix('./')}"

            src = f'<audio controls src="{url}" type="audio/mpeg"></audio>'
            if rec.get("gold_transcript") and not args.no_gold:
                src += f"<br><i>Gold transcript:</i><br>{esc(rec['gold_transcript'])}"

            # one text per system, spans grouped per system
            tgt, spans = {}, defaultdict(list)
            for t in rec["targets"]:
                system, text = t["system"], t["text"]
                if tgt.setdefault(system, text) != text:
                    print(f"WARNING: {rec['clip_id']}#{rec['segment']}: system {system} has "
                          f"different texts, keeping the first", file=sys.stderr)
                    continue
                if not t["span"] or t.get("span_start") is None:
                    # pure deletion, or a divergence flag with no span pinpointed:
                    # nothing to highlight, it stays in the instructions
                    continue
                s0 = t["span_start"] if text[t["span_start"]:t["span_end"]] == t["span"] else text.find(t["span"])
                if s0 < 0:
                    print(f"WARNING: span {t['span']!r} not in {rec['clip_id']}#{rec['segment']} "
                          f"({system})", file=sys.stderr)
                    continue
                # end_i is inclusive; the protocol has no severity buttons, so leave it unset
                spans[system].append({"start_i": s0, "end_i": s0 + len(t["span"]) - 1,
                                      "severity": None, "category": None})
                n_spans += 1

            item = {
                "item_id": f"{rec['clip_id']}#{rec['segment']:04d}",
                "src": src,
                "tgt": tgt,
                "instructions": instruction_html(rec, rec["targets"], args.show_asr, len(tgt) > 1),
                # extra keys are kept verbatim in the annotation logs
                "doc_id": doc_id, "clip_id": rec["clip_id"], "company": rec.get("company"),
                "filename": rec["filename"], "segment": rec["segment"],
                "segment_filename": rec["segment_filename"],
                "orig_start": rec.get("orig_start"), "orig_end": rec.get("orig_end"),
                "asr": rec.get("asr"), "gold_transcript": rec.get("gold_transcript"),
                "targets": rec["targets"],
            }
            if args.slim:
                for k in ("asr", "gold_transcript", "targets", "company", "filename",
                          "orig_start", "orig_end"):
                    item.pop(k, None)
            if spans and not args.no_prefill:
                item["error_spans"] = dict(spans)
            doc.append(item)
            n_items += 1
        if doc:
            documents.append(doc)

    if args.partition:
        tasks = [documents[i::args.users] for i in range(args.users)]
        tasks = [t for t in tasks if t]
    else:
        tasks = [documents for _ in range(args.users)]

    info = json.load(open(args.template, encoding="utf-8"))["info"]
    if args.shuffle != "keep":
        info["shuffle"] = args.shuffle == "on"
    if args.show_model_names != "keep":
        info["show_model_names"] = args.show_model_names == "on"
    campaign = {"info": info, "campaign_id": args.campaign_id, "data": tasks}
    json.dump(campaign, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if args.copy_assets:
        print(f"clips copied to {Path(args.copy_assets).resolve() / args.campaign_id}", file=sys.stderr)
    print(f"clips referenced as {assets_url}/<segment_filename>", file=sys.stderr)
    if missing:
        print(f"WARNING: {missing} clips missing -- run cut_clips.py first", file=sys.stderr)
    print(f"{len(documents)} documents, {n_items} items, {n_errors} errors, {n_spans} pre-filled spans, "
          f"{len(tasks)} task(s) {'partitioned' if args.partition else 'replicated'} -> {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
