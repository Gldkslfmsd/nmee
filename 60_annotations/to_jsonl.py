#!/usr/bin/env python3
"""Convert pearmut human annotations into the harmful-error format of 40_find_harmful_errors.py.

    python3 48_pearmut_to_harm.py --pearmut earnings-kimi-cs.jsonl --input aligned/all.jsonl \\
        -o out/human_{user}.jsonl

Pearmut input: one line per annotated page (one user, several items):
  {"user_id": ..., "item": [{"item_id": "<document>#<segment>", "src": "<audio ...><div>Gold transcript: ..",
                             "tgt": {"canary_asr": "...", "canary_cs": "..."}, "instructions": "<table>..."}],
   "annotation": [{"canary_asr": {"error_spans": [{"start_i", "end_i", "severity", "category"}], ...}}],
   "actions": [...]}
  start_i and end_i are character offsets, end_i inclusive.

Output: one line per segment with at least one human error span, in the format of
40_find_harmful_errors.py, with the same fields in the same order:
  document, dataset, src_language, audio, beg, end, segmented_by, asr, asr_system, gold_transcript,
  targets: [{tgt_lan, system, text, span, span_start, span_end (exclusive), intended, harm_types,
             harmfulness, explanation, error_source}],
  annotator: {model: "human:<user>", backend: "pearmut", level: "segment", annotated, shown}
Values the human annotation does not provide are null: intended, harmfulness, explanation,
error_source, and harm_types unless a category was chosen (beg, end, segmented_by without --input).
Unless --strict, pearmut-specific fields are added after the standard ones: "severity" (and
"omission") in every target, "user" in "annotator", and "pearmut_item_id", "suggestions" (the LLM
suggestion shown to the annotator, per system) and "human_other" (score, textfield, sliders) at the
end of the record.

OUTPUT.done lists every annotated segment, including those without any error, in the format of
40_find_harmful_errors.py, so that 45_analyze_agreement.py (without --input) compares humans and LLMs
on exactly the segments the human annotated.

--input (the aligned JSONL the LLMs annotated) provides beg, end, dataset, segmented_by, the languages
and the original audio path; segments are matched by the audio file name. Without it, beg and end are
unknown, the segments cannot be matched with LLM outputs, and no OUTPUT.done is written.

End markers: a span at or after the end of the text (start_i >= len(text)) is pearmut's marker past the
last character (in ESA: the [MISSING] token). In these annotations it is used on outputs that have no
other span, i.e. most likely to confirm "no harmful error", so it is dropped by default and counted;
--keep-end-markers keeps it as an empty span with "omission": true.

Several users: put {user} into -o to get one file per user (needed to compare annotators); otherwise
all users go into one file, with a warning.
"""
import argparse
import html
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from html.parser import HTMLParser

try:
    from prompts import HARM_TYPES
except ImportError:
    HARM_TYPES = ["False attribution", "Offensive", "Embarrassing or laughable",
                  "Derailing or contresens", "Safety, health or legal risk", "Other"]

ROW_FIELDS = {"suggested harmful error span": "span", "intended translation": "intended",
              "harm type": "harm_types", "likely source": "error_source",
              "explanation": "explanation", "harmfulness": "harmfulness"}


# ---------------------------------------------------------------- pearmut HTML

class _Table(HTMLParser):
    """Rows of cell texts of an HTML table."""

    def __init__(self):
        super().__init__()
        self.rows, self.cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            if self.rows:
                self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def split_harm_types(value):
    """"Offensive, Safety, health or legal risk" -> the known harm types it contains."""
    found = [h for h in HARM_TYPES if h in value]
    return found or [value]


def parse_instructions(text):
    """({system: language} from the header, {system: {field: value}}) of the suggestion table."""
    p = _Table()
    p.feed(text or "")
    if not p.rows:
        return {}, {}
    systems, lans = [], {}
    for h in p.rows[0][1:]:
        m = re.match(r"^(.*?)\s*\(([\w-]+)\)$", h)
        name = m.group(1) if m else h
        systems.append(name)
        if m:
            lans[name] = m.group(2).lower()
    suggestions = defaultdict(dict)
    for row in p.rows[1:]:
        if not row:
            continue
        label = row[0].rstrip(":").lower()
        field = ROW_FIELDS.get(label, re.sub(r"\W+", "_", label).strip("_"))
        for name, value in zip(systems, row[1:]):
            if value:
                suggestions[name][field] = split_harm_types(value) if field == "harm_types" else value
    return lans, dict(suggestions)


def parse_src(src):
    """(audio path, gold transcript) from the src HTML."""
    m = re.search(r'src="([^"]+)"', src or "")
    audio = html.unescape(m.group(1)) if m else None
    t = _Text()
    t.feed(src or "")
    text = " ".join(" ".join(t.parts).split())
    m = re.search(r"Gold transcript:\s*(.*)$", text)
    return audio, (m.group(1).strip() if m else None)


# ---------------------------------------------------------------- aligned input

def load_input(path):
    """{audio file name: record} of the aligned JSONL."""
    by_audio = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("audio"):
                    by_audio[os.path.basename(rec["audio"])] = rec
    return by_audio


def segment_key(rec):
    """The same key as in OUTPUT.done of 40_find_harmful_errors.py."""
    return f"{rec.get('document')}\t{rec.get('beg')}\t{rec.get('end')}"


# ---------------------------------------------------------------- conversion

def pick_asr(systems, lan_of, src_lan, wanted):
    if wanted:
        return wanted if wanted in systems else None
    same = [s for s in systems if src_lan and lan_of.get(s) == src_lan]
    named = [s for s in systems if "asr" in s.lower()]
    for group in ([s for s in same if s in named], same, named):
        if group:
            return group[0]
    return None


def convert_item(item, ann, user, by_audio, args, stats):
    """(segment record without targets filtered, list of targets, annotated systems)."""
    item_id = item.get("item_id", "")
    doc, _, seg_no = item_id.partition("#")
    audio_src, gold_html = parse_src(item.get("src"))
    lans, suggestions = parse_instructions(item.get("instructions"))
    tgt = item.get("tgt", {})
    systems = list(tgt)

    inp = None
    if by_audio is not None:
        inp = by_audio.get(os.path.basename(audio_src)) if audio_src else None
        if inp is None:
            stats["not found in --input"] += 1
    info = inp.get("systems_info", {}) if inp else {}
    lan_of = {s: info.get(s, {}).get("lan") or lans.get(s) for s in systems}
    src_lan = (inp or {}).get("src_language") or args.src_language
    asr = pick_asr(systems, lan_of, src_lan, args.asr_system)

    # the fields of the format of 40_find_harmful_errors.py, in its order; unknown values are null
    rec = OrderedDict()
    rec["document"] = inp.get("document") if inp else doc
    m = re.search(r"assets/([^/]+)/", audio_src or "")
    rec["dataset"] = inp.get("dataset") if inp else (m.group(1) if m else None)
    rec["src_language"] = src_lan
    rec["audio"] = inp.get("audio") if inp else audio_src
    rec["beg"] = inp.get("beg") if inp else None
    rec["end"] = inp.get("end") if inp else None
    rec["segmented_by"] = inp.get("segmented_by") if inp else None
    rec["asr"] = tgt.get(asr) if asr else None
    rec["asr_system"] = asr
    gold_system = next((s for s, v in info.items()
                        if v.get("is_human") and v.get("lan") == src_lan), None)
    gold = inp["text"].get(gold_system) if inp and gold_system else gold_html
    rec["gold_transcript"] = gold
    if inp:
        for s in systems:
            if inp.get("text", {}).get(s) is not None and inp["text"][s] != tgt[s]:
                stats["text differs from --input (spans refer to the pearmut text)"] += 1

    targets, human_other = [], {}
    for s in systems:
        text = tgt[s] or ""
        a = (ann or {}).get(s) or {}
        for sp in a.get("error_spans") or []:
            si, ei = sp.get("start_i"), sp.get("end_i")
            if si is None or ei is None:
                stats["span without offsets"] += 1
                continue
            si, ei = min(si, ei), max(si, ei)
            omission = si >= len(text)
            if omission:
                stats["end marker dropped" if not args.keep_end_markers else "end marker kept"] += 1
                if not args.keep_end_markers:
                    continue
                start = end = len(text)
            else:
                start, end = si, min(len(text), ei + 1)
            entry = OrderedDict()
            entry["tgt_lan"] = lan_of.get(s)
            entry["system"] = s
            entry["text"] = text
            entry["span"] = text[start:end]
            entry["span_start"] = start
            entry["span_end"] = end
            # not given by the human annotation
            entry["intended"] = None
            entry["harm_types"] = split_harm_types(sp["category"]) if sp.get("category") else None
            entry["harmfulness"] = None
            entry["explanation"] = None
            entry["error_source"] = None
            if not args.strict:  # pearmut-specific additions
                if sp.get("severity") is not None:
                    entry["severity"] = sp["severity"]
                if omission:
                    entry["omission"] = True
            targets.append(entry)
            stats[f"span in {s}"] += 1
        extra = {k: a.get(k) for k in ("score", "textfield", "sliders") if a.get(k) not in (None, {}, "")}
        if extra:
            human_other[s] = extra

    rec["targets"] = targets
    rec["annotator"] = OrderedDict(model=f"human:{user}", backend="pearmut", level="segment",
                                   annotated=systems,
                                   shown=systems + (["gold_transcript"] if gold else []))
    if not args.strict:  # pearmut-specific additions, after the standard fields
        rec["annotator"]["user"] = user
        rec["pearmut_item_id"] = item_id
        if suggestions:
            rec["suggestions"] = suggestions
        if human_other:
            rec["human_other"] = human_other
    return rec, systems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pearmut", required=True, help="pearmut annotation JSONL")
    ap.add_argument("--input", help="the aligned JSONL that was annotated (for beg/end, languages, ...)")
    ap.add_argument("-o", "--output", required=True,
                    help="output JSONL; {user} in the name writes one file per annotator")
    ap.add_argument("--asr-system", help="system that is the transcript (default: the one in the source "
                                         "language, preferably with 'asr' in its name)")
    ap.add_argument("--src-language", default="en", help="source language if not in --input (default en)")
    ap.add_argument("--strict", action="store_true",
                    help="only the fields of the format of 40_find_harmful_errors.py (no severity, "
                         "omission, annotator.user, pearmut_item_id, suggestions, human_other)")
    ap.add_argument("--keep-end-markers", action="store_true",
                    help="keep spans past the end of the text as empty omission spans")
    args = ap.parse_args()

    by_audio = load_input(args.input) if args.input else None
    if by_audio is None:
        print("WARNING: no --input: beg/end unknown, the output cannot be matched with LLM outputs "
              "and no .done file is written", file=sys.stderr)

    stats = Counter()
    per_user = defaultdict(OrderedDict)   # user -> {item_id: (record, systems)}
    with open(args.pearmut, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            page = json.loads(line)
            user = page.get("user_id") or "unknown"
            items, anns = page.get("item") or [], page.get("annotation") or []
            if len(items) != len(anns):
                print(f"WARNING: line {n}: {len(items)} items but {len(anns)} annotations; "
                      f"using the first {min(len(items), len(anns))}", file=sys.stderr)
            for item, ann in zip(items, anns):
                rec, systems = convert_item(item, ann, user, by_audio, args, stats)
                if item.get("item_id") in per_user[user]:
                    stats["annotated again (last kept)"] += 1
                per_user[user][item.get("item_id")] = (rec, systems)
                stats["segments annotated"] += 1

    users = list(per_user)
    if len(users) > 1 and "{user}" not in args.output:
        print(f"WARNING: {len(users)} annotators in one file; use -o '...{{user}}...' to compare them",
              file=sys.stderr)
        groups = {None: [x for u in users for x in per_user[u].values()]}
    else:
        groups = {u: list(per_user[u].values()) for u in users}

    for user, items in groups.items():
        path = args.output.replace("{user}", user) if user is not None else args.output
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        n_out = 0
        with open(path, "w", encoding="utf-8") as out:
            for rec, _ in items:
                if rec["targets"]:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
        msg = f"-> {path}: {n_out} segments with errors out of {len(items)} annotated"
        if by_audio is not None:
            known = [(rec, systems) for rec, systems in items if rec.get("beg") is not None]
            annotate = sorted({s for _, systems in known for s in systems})
            config = {"model": f"human:{user}" if user else "human", "backend": "pearmut",
                      "annotate": annotate, "source": os.path.basename(args.pearmut)}
            with open(path + ".done", "w", encoding="utf-8") as done:
                done.write("#config " + json.dumps(config, sort_keys=True, ensure_ascii=False) + "\n")
                for key in dict.fromkeys(segment_key(rec) for rec, _ in known):
                    done.write(key + "\n")
            msg += f"; {len(known)} segments in {path}.done"
        print(msg, file=sys.stderr)
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
