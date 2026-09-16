#!/bin/bash
# Run 01-rough-align.py on all 290 input portions (in parallel).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DATA_JSONL="input/no-more-embarrassing-errors-dominik/earnings-25/testset-segmented/data.jsonl"
OUT_DIR="input/no-more-embarrassing-errors-dominik/out"
DEST_DIR="15-roughaligned"
JOBS="${JOBS:-$(nproc)}"

# Prefer the local virtualenv if it exists.
PY="$HERE/venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

mkdir -p "$DEST_DIR"

# One line per recording id -> one rough-align.py job.
grep -o '"id": *"[^"]*"' "$DATA_JSONL" | sed 's/.*"id": *"\([^"]*\)".*/\1/' \
| xargs -P "$JOBS" -I{} "$PY" "$HERE/10-rough-align.py" {} \
    --out-dir "$OUT_DIR" --data-jsonl "$DATA_JSONL" --dest-dir "$DEST_DIR"

echo "Done: $(ls "$DEST_DIR" | wc -l) files in $DEST_DIR"
