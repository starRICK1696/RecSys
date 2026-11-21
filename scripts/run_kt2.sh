#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf 'Usage: %s MOVIELENS_DIR [OUTPUT_DIR]\n' "$0" >&2
    exit 2
fi

PROJECT_DIR=$(CDPATH= cd -P "$(dirname "$0")/.." && pwd)
DATA_DIR=$1
RECSYS_PYTHON=${RECSYS_PYTHON:-python3}
if [ "$#" -eq 2 ]; then
    OUTPUT_DIR=$2
else
    mkdir -p "$PROJECT_DIR/results/kt2"
    OUTPUT_DIR=$(mktemp -d "$PROJECT_DIR/results/kt2/pilot.XXXXXX")
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/recsys-mpl"
export PYTHONPATH="$(dirname "$PROJECT_DIR")${PYTHONPATH:+:$PYTHONPATH}"
"$RECSYS_PYTHON" -m RecSys.kt2_experiments --data "$DATA_DIR" --output "$OUTPUT_DIR" \
    --max-users 240 --max-items 600 --rank 16 --lambdas 1 10 --seeds 42 43 44
"$RECSYS_PYTHON" -m RecSys.plot_kt2 --input "$OUTPUT_DIR"
