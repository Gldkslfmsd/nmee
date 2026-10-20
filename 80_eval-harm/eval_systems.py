#!/usr/bin/env python3
"""Statistics per system from the output of same_harm.py.

    python3 eval_systems.py same_harm.fr.jsonl
    python3 eval_systems.py same_harm.cs.jsonl same_harm.fr.jsonl --tsv > stats.tsv

Input: one or more outputs of same_harm.py (one confirmed segment per line).

Every segment of the input contains a harmful error confirmed by a human in at least one system (the
"confirmed" system); the other systems were compared with it by the LLM ("same_harm" of each harm span:
true / false / null). Three tables are printed.

1. Counts per system (counted per segment; a segment with several harm spans counts once):
    confirmed  the human annotator confirmed a harmful error in this system
    compared   the LLM compared this system with a confirmed error
    same       ... and found the same error for at least one harm span
    different  ... and found none of them (all answers "no")
    unknown    ... no "yes", but at least one failed / unparsable answer
    same%      same / (same + different)
    harmful    confirmed or same: segments with a harmful error in this system, out of all segments
    spans yes, no, ?   the LLM answers per harm span; ? = failed request or unparsable answer

2. Harm rate per compared system: same / compared, with a 95% Wilson confidence interval. Unknown
   segments count as not harmful here. Note what it measures: the segments were selected because the
   confirmed system made a harmful error in them, so it is the rate at which a system repeats those
   errors, not its overall rate of harmful errors.

3. Tests of the difference between the systems, separately for every language, on the segments
   where the systems were compared (paired data: the same segments for all systems):
    - Cochran's Q test: do the harm rates of all compared systems of the language differ?
      Only segments where every system was compared and none is unknown.
    - pairwise exact McNemar tests (two-sided) with the Holm correction for multiple comparisons:
      segments where both systems were compared and neither is unknown. "only A" / "only B" are the
      discordant segments (the error repeated by one system but not the other); the concordant ones
      do not affect the test.
   With few segments the tests have little power: a non-significant result is not evidence that the
   systems are equal.

As a module: collect(records) -> (number of segments, {system: stats}); print_report(n, systems)
prints all three tables. same_harm.py uses it to print the statistics at the end of a run.
"""
import argparse
import itertools
import json
import math
import sys
from collections import Counter

COLUMNS = [("confirmed", "confirmed"), ("compared", "compared"), ("same", "same"),
           ("different", "different"), ("unknown", "unknown"), ("same%", None),
           ("harmful", None), ("spans yes", "span yes"), ("spans no", "span no"),
           ("spans ?", "span unknown")]
ALPHA = 0.05


# ---------------------------------------------------------------- collecting

def segment_outcome(verdicts):
    """True (same error for some span) / False (all "no") / None (unknown)."""
    if any(v is True for v in verdicts):
        return True
    if all(v is False for v in verdicts):
        return False
    return None


def collect(records):
    """(number of segments, {system: {"lan", "counts": Counter, "outcomes": {segment: True/False/None}}}).
    outcomes holds only the segments where the system was compared."""
    systems = {}

    def get(system, lan):
        entry = systems.setdefault(system, {"lan": lan, "counts": Counter(), "outcomes": {}})
        if entry["lan"] is None:
            entry["lan"] = lan
        return entry["counts"]

    n = 0
    for seg, rec in enumerate(records):
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
            c, outcome = systems[system]["counts"], segment_outcome(vs)
            systems[system]["outcomes"][seg] = outcome
            c["compared"] += 1
            c["same" if outcome is True else "different" if outcome is False else "unknown"] += 1
        for system in confirmed | {s for s, vs in verdicts.items() if segment_outcome(vs) is True}:
            systems[system]["counts"]["harmful"] += 1
    return n, systems


def by_language(systems):
    """[(lan, [system, ...])] of the compared systems, sorted."""
    langs = {}
    for system, entry in sorted(systems.items()):
        if entry["counts"]["compared"]:
            langs.setdefault(entry["lan"] or "-", []).append(system)
    return sorted(langs.items())


# ---------------------------------------------------------------- statistics

def wilson(k, n, z=1.96):
    """95% Wilson score interval of the proportion k/n."""
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test: binomial test of the discordant pairs b, c with p = 0.5."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def gammaincc(a, x):
    """Regularized upper incomplete gamma function Q(a, x) (Numerical Recipes, series / fraction)."""
    if x <= 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # series for P(a, x)
        ap, term = a, 1 / a
        total = term
        for _ in range(10000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1 - total * math.exp(log_front))
    tiny = 1e-300  # continued fraction for Q(a, x)
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(log_front) * h


def chi2_sf(x, df):
    return gammaincc(df / 2, x / 2)


def cochran_q(rows):
    """(Q, df, p) of Cochran's Q test; rows: [[0/1 per system], ...] (complete segments only)."""
    k = len(rows[0]) if rows else 0
    if k < 2 or not rows:
        return None, k - 1, None
    col = [sum(r[j] for r in rows) for j in range(k)]
    row = [sum(r) for r in rows]
    total = sum(row)
    denom = k * total - sum(r * r for r in row)
    if denom == 0:  # every segment has all systems equal: no information
        return 0.0, k - 1, 1.0
    q = (k - 1) * (k * sum(c * c for c in col) - total * total) / denom
    return q, k - 1, chi2_sf(q, k - 1)


def holm(pvalues):
    """Holm-adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------- tables

def counts_rows(n, systems):
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


def harm_rate_rows(systems):
    out = [["system", "lan", "compared", "same", "harm-rate%", "95% CI"]]
    for lan, names in by_language(systems):
        for system in sorted(names, key=lambda s: (-systems[s]["counts"]["same"]
                                                   / systems[s]["counts"]["compared"], s)):
            c = systems[system]["counts"]
            lo, hi = wilson(c["same"], c["compared"])
            out.append([system, lan, str(c["compared"]), str(c["same"]),
                        f"{100 * c['same'] / c['compared']:.1f}", f"{100 * lo:.1f}-{100 * hi:.1f}"])
    return out


def test_report(systems):
    """(lines about Cochran's Q, pairwise McNemar rows) for every language."""
    lines = []
    pairs = [["lan", "system A", "system B", "n", "rate A%", "rate B%", "only A", "only B", "p", "p Holm", ""]]
    for lan, names in by_language(systems):
        if len(names) < 2:
            lines.append(f"{lan}: only one compared system ({names[0]}), nothing to test")
            continue
        out = {s: systems[s]["outcomes"] for s in names}
        segs = set.intersection(*(set(o) for o in out.values()))
        complete = [seg for seg in sorted(segs) if all(out[s][seg] is not None for s in names)]
        q, df, p = cochran_q([[int(out[s][seg]) for s in names] for seg in complete])
        if p is None:
            lines.append(f"{lan}: Cochran's Q: no segment where all {len(names)} systems were compared")
        else:
            verdict = "the systems differ" if p < ALPHA else "no significant difference"
            lines.append(f"{lan}: Cochran's Q = {q:.3f}, df = {df}, p = {p:.4g} on {len(complete)} "
                         f"segments with all {len(names)} systems -> {verdict} (alpha {ALPHA})")
        rows, pvals = [], []
        for a, b in itertools.combinations(names, 2):
            shared = [seg for seg in sorted(set(out[a]) & set(out[b]))
                      if out[a][seg] is not None and out[b][seg] is not None]
            only_a = sum(1 for seg in shared if out[a][seg] and not out[b][seg])
            only_b = sum(1 for seg in shared if out[b][seg] and not out[a][seg])
            rate_a = sum(1 for seg in shared if out[a][seg])
            rate_b = sum(1 for seg in shared if out[b][seg])
            pct = (lambda k: f"{100 * k / len(shared):.1f}" if shared else "-")
            rows.append([lan, a, b, str(len(shared)), pct(rate_a), pct(rate_b),
                         str(only_a), str(only_b)])
            pvals.append(mcnemar_exact(only_a, only_b))
        for row, p_raw, p_adj in zip(rows, pvals, holm(pvals)):
            pairs.append(row + [f"{p_raw:.4g}", f"{p_adj:.4g}", "*" if p_adj < ALPHA else ""])
    return lines, pairs


def print_rows(table, file, tsv, left=2):
    if tsv:
        for row in table:
            print("\t".join(row), file=file)
        return
    widths = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    for k, row in enumerate(table):
        print("  ".join(cell.ljust(w) if i < left else cell.rjust(w)
                        for i, (cell, w) in enumerate(zip(row, widths))).rstrip(), file=file)
        if k == 0:
            print("  ".join("-" * w for w in widths), file=file)


def print_table(n, systems, file=sys.stdout, tsv=False):
    """Table 1: counts per system."""
    if not tsv:
        print(f"{n} segments with a human-confirmed harmful error", file=file)
    print_rows(counts_rows(n, systems), file, tsv)


def print_report(n, systems, file=sys.stdout, tsv=False):
    """All three tables."""
    print_table(n, systems, file, tsv)
    print(file=file)
    if not tsv:
        print("Harm rate = same / compared (unknown counted as not the same), 95% Wilson CI", file=file)
    print_rows(harm_rate_rows(systems), file, tsv)
    print(file=file)
    lines, pairs = test_report(systems)
    if not tsv:
        print("Tests of the difference between systems (paired, per language)", file=file)
        for line in lines:
            print(line, file=file)
    if len(pairs) > 1:
        if not tsv:
            print(f"\nPairwise exact McNemar tests, Holm-corrected (* = p Holm < {ALPHA}); "
                  f"only A / only B = segments where only that system repeats the error", file=file)
        print_rows(pairs, file, tsv, left=3)


def read(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="output(s) of same_harm.py")
    ap.add_argument("--tsv", action="store_true", help="tab-separated tables (no titles or test summary)")
    args = ap.parse_args()
    n, systems = collect(read(args.files))
    print_report(n, systems, tsv=args.tsv)


if __name__ == "__main__":
    main()