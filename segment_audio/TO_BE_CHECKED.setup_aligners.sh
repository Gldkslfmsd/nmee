#!/usr/bin/env bash
# Add and build the aligners used by add_and_align_sentences.py as git submodules under third_party/:
#
#   third_party/vecalign    https://github.com/thompsonb/vecalign   (--align vecalign)
#   third_party/mweralign   https://github.com/mjpost/mweralign     (--align mwer)
#
# Vecalign is not on PyPI and has a Cython extension, so it is built in place here; the script picks it
# up from the submodule automatically (override with --vecalign-dir / --mweralign-dir).
#
# Usage:
#   ./setup_aligners.sh              # both
#   ./setup_aligners.sh vecalign     # only one of: vecalign, mweralign
#
# Needs: git, a C/C++ compiler, Python headers (e.g. apt install build-essential python3-dev), cmake
# for mweralign, and: pip install cython numpy pybind11 sentence-transformers mosestokenizer
set -euo pipefail

cd "$(dirname "$0")"
WHAT=${1:-all}
mkdir -p third_party

add_submodule() {  # name url
    local name=$1 url=$2 dir="third_party/$1"
    if [[ -d "$dir/.git" || -f "$dir/.git" ]]; then
        echo "== $name: already there, updating"
        git -C "$dir" pull --ff-only || true
    elif git rev-parse --git-dir > /dev/null 2>&1; then
        echo "== $name: adding as a git submodule"
        git submodule add "$url" "$dir" || git submodule update --init -- "$dir"
    else
        echo "== $name: not a git repository, cloning instead"
        git clone "$url" "$dir"
    fi
}

if [[ "$WHAT" == "all" || "$WHAT" == "vecalign" ]]; then
    add_submodule vecalign https://github.com/thompsonb/vecalign.git
    echo "== vecalign: building the Cython extension in place"
    ( cd third_party/vecalign && python setup.py build_ext --inplace )
    python - <<'PY'
import sys
sys.path.insert(0, "third_party/vecalign")
from vecalign.dp_utils import vecalign  # noqa: F401
print("== vecalign: OK")
PY
fi

if [[ "$WHAT" == "all" || "$WHAT" == "mweralign" ]]; then
    add_submodule mweralign https://github.com/mjpost/mweralign.git
    echo "== mweralign: building (pip install -e, needs cmake + pybind11)"
    pip install -e third_party/mweralign
    python -c "from mweralign import align_texts; print('== mweralign: OK')"
fi

cat <<'EOF'

Done. To fetch the submodules on another machine:

    git submodule update --init --recursive
    ./setup_aligners.sh

Both repositories are pinned by the submodule commit, so alignments stay reproducible.
EOF
