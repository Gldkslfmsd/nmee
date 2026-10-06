#!/usr/bin/env python3
"""Agreement between runs of 40_find_harmful_errors.py on detecting harmful errors.

    python3 45_analyze_agreement.py --input first100.jsonl out/Gemma4_cs.jsonl out/Kimi_de.jsonl \\
        [more runs ...] [--heatmap kappa.png]

The output of 40_find_harmful_errors.py contains only segments with at least one flagged error.
With --input (the aligned JSONL that was annotated, one segment per line), every segment of it is
assumed to have been annotated by every run: a segment a run did not flag is a negative for every system
that run annotated. Without --input, the segments listed in OUTPUT.done next to each run are used (the
ones annotated by all runs); without those either, only the segments flagged by at least one run are
known, and the negatives are not meaningful (with a warning).

Every (run, system) combination is an entry, e.g. Gemma4_cs:canary_asr, Gemma4_cs:canary_cs,
Kimi_de:canary_asr, Kimi_de:canary_de. For every pair of entries: the 2x2 confusion matrix of "the
segment has a flagged error" (yes = flagged, no = not flagged), Cohen's kappa, positive agreement, and
how often one confirms the other. The kappas are summarised in a matrix (and optionally a heatmap).
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

SYSTEM_KEYWORDS = {"all", "targets", "human"}  # --annotate keywords of 40_find_harmful_errors.py


# ---------------------------------------------------------------- loading

def segment_key(rec):
    """The same key as in OUTPUT.done."""
    return f"{rec.get('document')}\t{rec.get('beg')}\t{rec.get('end')}"


def load_input(path):
    """(segment keys in input order, without duplicates; all system names in the input)."""
    keys, systems = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                keys.append(segment_key(rec))
                systems.update(rec.get("systems_info", {}))
    unique = list(dict.fromkeys(keys))
    if len(unique) < len(keys):
        print(f"note: {len(keys) - len(unique)} duplicate segments in {path} are counted once",
              file=sys.stderr)
    return unique, systems


def load_run(path, known_systems=None):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                recs[segment_key(rec)] = rec
    done, config = None, {}
    done_path = path + ".done"
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        if lines and lines[0].startswith("#config "):
            config = json.loads(lines[0][len("#config "):])
            lines = lines[1:]
        done = {l for l in lines if l.strip()}
    systems = set()
    flags = defaultdict(list)  # (segment key, system) -> [target entries]
    for key, rec in recs.items():
        systems.update(rec.get("annotator", {}).get("annotated", []))
        for t in rec.get("targets", []):
            systems.add(t["system"])
            flags[(key, t["system"])].append(t)
    # systems that were annotated but never flagged are known only from the run's settings
    for s in config.get("annotate") or []:
        if s in SYSTEM_KEYWORDS:
            continue
        if known_systems is None or s in known_systems:
            systems.add(s)
    return {"name": Path(path).stem, "path": path, "recs": recs, "done": done,
            "config": config, "systems": systems, "flags": flags}


def harm_of(run, key, system):
    """0 if the run flagged nothing in this system and segment, else the maximum harmfulness."""
    hs = [t.get("harmfulness") or 1 for t in run["flags"].get((key, system), [])]
    return max(hs) if hs else 0


def pct(x, n):
    return f"{100 * x / n:.1f}%" if n else "-"


def num(v):
    return "-" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------- reports

def report_models(runs, keys):
    print("=" * 100)
    print("RUNS")
    print("=" * 100)
    for r in runs:
        cfg = r["config"]
        extra = (f"  [model {cfg.get('model')}, group size {cfg.get('group_size', 1)}]"
                 if cfg else "")
        print(f"\n{r['name']}  ({r['path']}){extra}")
        for system in sorted(r["systems"]):
            hs = [harm_of(r, k, system) for k in keys]
            flagged = sum(1 for h in hs if h)
            spans = sum(len(r["flags"].get((k, system), [])) for k in keys)
            dist = Counter(h for h in hs if h)
            print(f"  {system:16} {flagged:5}/{len(keys)} segments flagged ({pct(flagged, len(keys))}), "
                  f"{spans:5} spans | max harmfulness per segment: "
                  + "  ".join(f"{h}:{dist[h]}" for h in range(1, 6)))


def binary_kappa(both, only_a, only_b, neither):
    n = both + only_a + only_b + neither
    if not n:
        return None
    po = (both + neither) / n
    pa, pb = (both + only_a) / n, (both + only_b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def report_pairs(entries, keys):
    """2x2 detection matrix and kappa for every pair of entries; returns {(i, j): kappa}."""
    n = len(keys)
    kappas = {}
    for i, j in combinations(range(len(entries)), 2):
        a, b = entries[i], entries[j]
        both = len(a["flagged"] & b["flagged"])
        only_a = len(a["flagged"] - b["flagged"])
        only_b = len(b["flagged"] - a["flagged"])
        neither = n - both - only_a - only_b
        k = binary_kappa(both, only_a, only_b, neither)
        f1 = 2 * both / (2 * both + only_a + only_b) if both + only_a + only_b else None
        kappas[(i, j)] = kappas[(j, i)] = k

        print("\n" + "=" * 100)
        print(f"A = {a['label']}   vs   B = {b['label']}   ({n} segments)")
        print("=" * 100)
        print("  1) detection (yes = flagged, no = not flagged)")
        print("     rows: A, columns: B")
        print(" " * 13 + f"{'yes':>8}{'no':>8}")
        print(" " * 5 + f"{'yes':>8}{both:8}{only_a:8}")
        print(" " * 5 + f"{'no':>8}{only_b:8}{neither:8}")
        print(f"     kappa {num(k)} | positive agreement {num(f1)} | "
              f"B confirms A: {pct(both, both + only_a)} | A confirms B: {pct(both, both + only_b)}")
    return kappas


def report_kappa_matrix(entries, kappas):
    labels = [e["label"] for e in entries]
    width = max(len(l) for l in labels)
    print("\n" + "=" * 100)
    print("KAPPA MATRIX: detection (yes = flagged, no = not flagged)")
    print("=" * 100)
    print(" " * (width + 5) + "".join(f"{j + 1:>7}" for j in range(len(entries))))
    for i, label in enumerate(labels):
        cells = []
        for j in range(len(entries)):
            if i == j:
                cells.append(f"{'1':>7}")
            else:
                v = kappas.get((i, j))
                cells.append(f"{'-':>7}" if v is None else f"{v:7.2f}")
        print(f"{i + 1:>3}  {label:<{width}}" + "".join(cells))
    print("\n  columns are numbered as the rows; '-' = undefined (an entry never flags)")


def plot_heatmap(entries, kappas, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"WARNING: matplotlib is not installed, no heatmap written to {path}", file=sys.stderr)
        return
    labels = [e["label"] for e in entries]
    m = len(labels)
    data = [[1.0 if i == j else (kappas.get((i, j)) if kappas.get((i, j)) is not None
                                 else float("nan")) for j in range(m)] for i in range(m)]
    size = max(4, 0.6 * m + 2)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(m))
    ax.set_yticks(range(m))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(m):
        for j in range(m):
            v = data[i][j]
            text = "-" if v != v else f"{v:.2f}"  # v != v: NaN
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="black" if v == v and v > 0.6 else "white")
    fig.colorbar(im, ax=ax, label="Cohen's kappa (detection)")
    ax.set_title("Agreement on flagged segments")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nheatmap -> {path}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="outputs of 40_find_harmful_errors.py")
    ap.add_argument("--input",
                    help="the aligned JSONL that was annotated (input of 40_find_harmful_errors.py); "
                         "all its segments are assumed annotated by every run, not flagged = negative")
    ap.add_argument("--heatmap", help="write the kappa matrix as a heatmap image (e.g. kappa.png; "
                                      "needs matplotlib)")
    args = ap.parse_args()

    input_keys, input_systems = (load_input(args.input) if args.input else (None, None))
    runs = [load_run(p, input_systems) for p in args.files]
    names = [r["name"] for r in runs]
    if len(set(names)) < len(names):  # same file name in different directories
        for r in runs:
            r["name"] = r["path"]

    if input_keys is not None:
        keys = input_keys
        keyset = set(keys)
        for r in runs:
            outside = set(r["recs"]) - keyset
            if outside:
                print(f"WARNING: {len(outside)} flagged segments of {r['name']} are not in "
                      f"{args.input} (different input or segmentation?); ignored", file=sys.stderr)
            unknown = r["systems"] - input_systems
            if unknown:
                print(f"WARNING: systems of {r['name']} not in {args.input}: "
                      f"{', '.join(sorted(unknown))}", file=sys.stderr)
        print(f"{len(runs)} runs, {len(keys)} segments in {args.input} "
              f"(all assumed annotated by every run; not flagged = negative)")
    elif all(r["done"] is not None for r in runs):
        keyset = set.intersection(*(r["done"] for r in runs))
        dropped = set.union(*(r["done"] for r in runs)) - keyset
        if dropped:
            print(f"note: {len(dropped)} segments not annotated by all runs are ignored",
                  file=sys.stderr)
        keys = sorted(keyset)
        print(f"{len(runs)} runs, {len(keys)} segments annotated by all of them")
    else:
        missing = [r["name"] for r in runs if r["done"] is None]
        print(f"WARNING: no --input and no .done file for {', '.join(missing)}: segments without "
              f"any flag are unknown, using only the segments flagged by at least one run -- the "
              f"'no/no' cell and the kappas are not meaningful", file=sys.stderr)
        keyset = set.union(*(set(r["recs"]) for r in runs))
        keys = sorted(keyset)
        print(f"{len(runs)} runs, {len(keys)} segments flagged by at least one of them")
    if not keys:
        sys.exit("no segments")
    keyset = set(keys)

    report_models(runs, keys)

    entries = [{"label": f"{r['name']}:{s}",
                "flagged": {k for (k, sys_) in r["flags"] if sys_ == s and k in keyset}}
               for r in runs for s in sorted(r["systems"])]
    if len(entries) < 2:
        return
    kappas = report_pairs(entries, keys)
    report_kappa_matrix(entries, kappas)
    if args.heatmap:
        plot_heatmap(entries, kappas, args.heatmap)


if __name__ == "__main__":
    main()