#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 0 ]; then
  echo "Usage: $0 [constraint] [benchmark_name] [userdir]"
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-./venv/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  echo "Create the AMLB virtual environment first, or set PYTHON_BIN explicitly."
  exit 1
fi

CONSTRAINT=${1:-flaml_paper_30m}
BENCHMARK_NAME=${2:-flaml_paper_39}
USERDIR=${3:-examples/custom_myautoml}

for FRAMEWORK in accelera_automl flaml autosklearn TPOT; do
  "${PYTHON_BIN}" runbenchmark.py "${FRAMEWORK}" "${BENCHMARK_NAME}" "${CONSTRAINT}" \
    -u "${USERDIR}" \
    -o "results_${BENCHMARK_NAME}_${FRAMEWORK}_${CONSTRAINT}" \
    --resume
done
