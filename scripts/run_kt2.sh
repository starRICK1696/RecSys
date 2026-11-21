#!/bin/sh
# Run from the RecSys directory. Optional first argument: MovieLens directory.
set -eu
DATA_DIR=${1:-/Users/dmitrijsuevalov/Downloads/ml-1m}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/recsys-mpl"
export PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python -m RecSys.kt2_experiments --data "$DATA_DIR" --output output/kt2/pilot_v2 --max-users 240 --max-items 600 --rank 16 --lambdas 1 10 --seeds 42 43 44
.venv/bin/python -m RecSys.plot_kt2 --input output/kt2/pilot_v2
