"""
Shared code for downloading Hugging Face speech datasets stored as
<config>/<split>-XXXXX-of-YYYYY.parquet shards with an embedded audio{bytes,path}
column, and converting them into audio files + JSONL manifests.

Output layout (same for every dataset):
    OUT/
      audio/<config>/<split>/<group>/<id>.<ext>          original bytes, no re-encoding
      manifests/<config>/<split>.jsonl                   one line per segment
      manifests/<config>/.shards/<split>/<shard>.jsonl   per-shard parts (resume bookkeeping)
      manifests/all.jsonl                                optional (--merge-all)

A dataset is described by a DatasetSpec whose `row_fn` turns one parquet row
into (group, id, record). Everything else (listing, downloading, parallelism,
resuming, merging) lives here.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SPLITS = ("train", "validation", "test")
SHARD_RE = re.compile(r"^(?P<config>[^/]+)/(?P<split>train|validation|test)-(?P<idx>\d+)-of-(?P<n>\d+)\.parquet$")

# row (without the audio column), config, split -> (group dir name, file stem, extra/overridden record fields)
RowFn = Callable[[dict, str, str], tuple[str, str, dict]]


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    row_fn: RowFn                      # must be a module-level function (pickled to worker processes)
    sampling_rate: int = 16000
    exclude_configs: frozenset = frozenset()


# --------------------------------------------------------------------------- helpers

def sniff_ext(b: bytes, path_hint: str | None) -> str:
    if path_hint:
        ext = os.path.splitext(path_hint)[1].lower()
        if ext in {".wav", ".flac", ".mp3", ".ogg", ".opus"}:
            return ext
    if b[:4] == b"RIFF":
        return ".wav"
    if b[:4] == b"fLaC":
        return ".flac"
    if b[:4] == b"OggS":
        return ".ogg"
    if b[:3] == b"ID3" or b[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    return ".wav"


def safe(s) -> str:
    """Filesystem-safe name component."""
    return re.sub(r"[^\w.\-]", "_", str(s))


def list_shards(repo_id: str, revision: str | None, exclude: frozenset = frozenset()):
    """Return {(config, split): [repo filenames]} for all parquet shards in the repo."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    shards = defaultdict(list)
    for f in files:
        m = SHARD_RE.match(f)
        if m and m["config"] not in exclude:
            shards[(m["config"], m["split"])].append(f)
    for k in shards:
        shards[k].sort()
    return dict(shards)


# --------------------------------------------------------------------------- conversion

def convert_parquet(parquet_path: Path, spec: DatasetSpec, config: str, split: str, out: Path,
                    part_manifest: Path, absolute_paths: bool, overwrite_audio: bool) -> int:
    """Write audio files + a per-shard JSONL. Returns number of rows written."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    tmp = part_manifest.with_suffix(".jsonl.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        n = _write_shard(pf, spec, config, split, out, tmp, parquet_path, absolute_paths, overwrite_audio)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(part_manifest)  # atomic rename marks the shard as done
    return n


def _write_shard(pf, spec, config, split, out, tmp, parquet_path, absolute_paths, overwrite_audio) -> int:
    n = 0
    made_dirs = set()
    with open(tmp, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                audio = row.pop("audio", None) or {}
                b = audio.get("bytes")
                group, stem, fields = spec.row_fn(row, config, split)
                if not b:
                    print(f"[warn] no audio bytes for {stem} in {parquet_path.name}", file=sys.stderr)
                    continue
                d = out / "audio" / config / split / safe(group or "unknown")
                if d not in made_dirs:
                    d.mkdir(parents=True, exist_ok=True)
                    made_dirs.add(d)
                fpath = d / f"{safe(stem)}{sniff_ext(b, audio.get('path'))}"
                if overwrite_audio or not fpath.exists():
                    with open(fpath, "wb") as fa:
                        fa.write(b)
                rec = {
                    "audio_filepath": str(fpath.resolve() if absolute_paths else fpath.relative_to(out)),
                    "sampling_rate": spec.sampling_rate,
                    "split": split,
                    **row,
                    **fields,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def process_shard(spec: DatasetSpec, filename: str, config: str, split: str, out: str, cache: str,
                  keep_parquet: bool, absolute_paths: bool, overwrite_audio: bool,
                  revision: str | None) -> tuple[str, int]:
    from huggingface_hub import hf_hub_download

    out_p = Path(out)
    part = out_p / "manifests" / config / ".shards" / split / (Path(filename).stem + ".jsonl")
    if part.exists():
        return filename, -1
    local = hf_hub_download(spec.repo_id, filename, repo_type="dataset", local_dir=cache, revision=revision)
    n = convert_parquet(Path(local), spec, config, split, out_p, part, absolute_paths, overwrite_audio)
    if not keep_parquet:
        try:
            os.remove(local)
        except OSError:
            pass
    return filename, n


def merge_manifests(out: Path, config: str, split: str) -> int:
    parts_dir = out / "manifests" / config / ".shards" / split
    parts = sorted(parts_dir.glob("*.jsonl")) if parts_dir.is_dir() else []
    if not parts:
        return 0
    dst = out / "manifests" / config / f"{split}.jsonl"
    n = 0
    with open(dst, "w", encoding="utf-8") as fout:
        for p in parts:
            with open(p, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    n += 1
    return n


# --------------------------------------------------------------------------- CLI plumbing

def add_common_args(ap: argparse.ArgumentParser, default_out: str) -> None:
    ap.add_argument("--out", default=default_out, help=f"output root directory (default: {default_out})")
    ap.add_argument("--cache", default=None, help="where parquet shards are downloaded (default: OUT/_parquet)")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--max-shards", type=int, default=None, help="per (config, split); for quick tests")
    ap.add_argument("--workers", type=int, default=4, help="parallel shards (download + convert)")
    ap.add_argument("--keep-parquet", action="store_true", help="don't delete parquet shards after conversion")
    ap.add_argument("--absolute-paths", action="store_true", help="absolute audio paths in JSONL")
    ap.add_argument("--overwrite-audio", action="store_true")
    ap.add_argument("--merge-all", action="store_true", help="also write manifests/all.jsonl")
    ap.add_argument("--revision", default=None, help="dataset git revision (pin for reproducibility)")
    ap.add_argument("--list", action="store_true", help="only list available configs/splits/shard counts")


def print_listing(shards: dict, label: Callable[[str], str] = lambda c: "") -> None:
    for (c, s), fs in sorted(shards.items()):
        print(f"{c:20s} {label(c):4s} {s:11s} {len(fs):4d} shards")


def run(spec: DatasetSpec, args: argparse.Namespace, shards: dict, configs: set[str]) -> None:
    """Download + convert the selected configs, then merge manifests."""
    out = Path(args.out)
    cache = Path(args.cache) if args.cache else out / "_parquet"

    jobs = [(f, c, s)
            for (c, s), fs in sorted(shards.items()) if c in configs and s in args.splits
            for f in fs[: args.max_shards]]
    if not jobs:
        sys.exit("nothing selected")

    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} shards to process -> {out.resolve()}", file=sys.stderr)

    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, spec, f, c, s, str(out), str(cache), args.keep_parquet,
                          args.absolute_paths, args.overwrite_audio, args.revision): f for f, c, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fname, n = fut.result()
                print(f"[{i}/{len(jobs)}] {fname}: {'skipped (done)' if n < 0 else f'{n} rows'}", file=sys.stderr)
            except Exception as e:  # keep going; rerun to retry failed shards
                failed += 1
                print(f"[{i}/{len(jobs)}] FAILED {futs[fut]}: {e!r}", file=sys.stderr)

    all_f = open(out / "manifests" / "all.jsonl", "w", encoding="utf-8") if args.merge_all else None
    try:
        for c in sorted({c for _, c, _ in jobs}):
            for s in args.splits:
                n = merge_manifests(out, c, s)
                if n:
                    print(f"manifests/{c}/{s}.jsonl: {n} lines", file=sys.stderr)
                    if all_f:
                        with open(out / "manifests" / c / f"{s}.jsonl", encoding="utf-8") as fin:
                            all_f.writelines(fin)
    finally:
        if all_f:
            all_f.close()

    if failed:
        sys.exit(f"{failed} shard(s) failed; rerun the same command to retry them")
