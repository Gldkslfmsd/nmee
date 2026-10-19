#!/usr/bin/env python3
"""Statistics per system from the output of same_harm.py.

    python3 eval_systems.py same_harm.fr.jsonl
    python3 eval_systems.py same_harm.cs.jsonl same_harm.fr.jsonl --tsv > stats.tsv

Input: one or more outputs of same_harm.py (one confirmed segment per line).

Every segment of the input contains a harmful error confirmed by a human in at least one system (the
"confirmed" system); the other systems were compared with it by the LLM ("same_harm" of each harm span:
true / false / null). Per system, the table has:

  segments (counted per segment, a segment with several harm spans counts once):
    confirmed  the human annotator confirmed a harmful error in this system
    compared   the LLM compared this system with a confirmed error
    same       ... and found the same error for at least one harm span
    different  ... and found none of them (all answers "no")
    unknown    ... no "yes", but at least one failed / unparsable answer
    same%      same / (same + different)
    harmful    confirmed or same: segments with a harmful error in this system, out of all segments
  spans (counted per confirmed harm span):
    yes, no, ?  the LLM answers; ? = failed request or unparsable answer

As a module: collect(records) -> (number of segments, {system: stats}); print_table(...) prints it.
same_harm.py uses it to print the statistics at the end of a run.
"""
import argparse
import json
import sys
from collections import Counter

COLUMNS = [("confirmed", "confirmed"), ("compared", "compared"), ("same", "same"),
           ("different", "different"), ("unknown", "unknown"), ("same%", None),
           ("harmful", None), ("spans yes", "span yes"), ("spans no", "span no"),
           ("spans ?", "span unknown")]


def collect(records):
    """(number of segments, {system: {"lan": ..., "counts": Counter}})."""
    systems = {}

    def get(system, lan):
        entry = systems.setdefault(system, {"lan": lan, "counts": Counter()})
        if entry["lan"] is None:
            entry["lan"] = lan
        return entry["counts"]

    n = 0
    for rec in records:
        n += 1
        confirmed, verdicts = set(), {}  # verdicts: system -> [True/False/None] in this segment
        for t in rec.get("targets", []):
            spans = t.get("harm_spans") or []
            if spans:
                confirmed.add(t.get("system"))
                get(t.get("system"), t.get("tgt_lan"))
            for sp in spans:
                for system, v in (sp.get("same_harm") or {}).items():
                    c = get(system, t.get("tgt_lan"))
                    c["span yes" if v is True else "span no" if v is False else "span unknown"] += 1
                    verdicts.setdefault(system, []).append(v)
        for system in confirmed:
            systems[system]["counts"]["confirmed"] += 1
        for system, vs in verdicts.items():
            c = systems[system]["counts"]
            c["compared"] += 1
            if any(v is True for v in vs):
                c["same"] += 1
            elif all(v is False for v in vs):
                c["different"] += 1
            else:
                c["unknown"] += 1
        for system in confirmed | {s for s, vs in verdicts.items() if any(v is True for v in vs)}:
            systems[system]["counts"]["harmful"] += 1
    return n, systems


def rows(n, systems):
    """[header, row, ...] as strings, systems sorted by language and name."""
    out = [["system", "lan"] + [label for label, _ in COLUMNS]]
    for system, entry in sorted(systems.items(), key=lambda kv: ((kv[1]["lan"] or ""), kv[0])):
        c = entry["counts"]
        row = [system, entry["lan"] or "-"]
        for label, key in COLUMNS:
            if label == "same%":
                decided = c["same"] + c["different"]
                row.append(f"{100 * c['same'] / decided:.1f}" if decided else "-")
            elif label == "harmful":
                row.append(f"{c['harmful']}/{n}")
            else:
                row.append(str(c[key]))
        out.append(row)
    return out


def print_table(n, systems, file=sys.stdout, tsv=False):
    table = rows(n, systems)
    if tsv:
        for row in table:
            print("\t".join(row), file=file)
        return
    print(f"{n} segments with a human-confirmed harmful error", file=file)
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    for k, row in enumerate(table):
        print("  ".join(cell.ljust(w) if i < 2 else cell.rjust(w)
                        for i, (cell, w) in enumerate(zip(row, widths))), file=file)
        if k == 0:
            print("  ".join("-" * w for w in widths), file=file)


def read(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of same_harm.py")
    ap.add_argument("--tsv", action="store_true", help="tab-separated output (no title line)")
    args = ap.parse_args()
    n, systems = collect(read(args.files))
    print_table(n, systems, tsv=args.tsv)


if __name__ == "__main__":
    main()
