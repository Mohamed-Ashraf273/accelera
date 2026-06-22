#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 FRAMEWORK [constraint] [benchmark_name] [userdir]"
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-./venv/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  echo "Create the AMLB virtual environment first, or set PYTHON_BIN explicitly."
  exit 1
fi

FRAMEWORK=${1}
CONSTRAINT=${2:-flaml_paper_30m}
BENCHMARK_NAME=${3:-flaml_paper_39}
USERDIR=${4:-examples/custom_myautoml}

"${PYTHON_BIN}" runbenchmark.py "${FRAMEWORK}" "${BENCHMARK_NAME}" "${CONSTRAINT}" \
  -u "${USERDIR}" \
  -o "results_${BENCHMARK_NAME}_${FRAMEWORK}_${CONSTRAINT}" \
  --resume
