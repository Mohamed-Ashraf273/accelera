#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 FRAMEWORK /path/to/pmlb [constraint] [benchmark_name] [userdir]"
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-./venv/bin/python}
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python executable not found: ${PYTHON_BIN}"
  echo "Create the AMLB virtual environment first, or set PYTHON_BIN explicitly."
  exit 1
fi

FRAMEWORK=${1}
PMLB_ROOT=${2}
CONSTRAINT=${3:-pmlb_30m}
BENCHMARK_NAME=${4:-pmlb_regression}
USERDIR=${5:-examples/custom_myautoml}
BENCHMARK_FILE="${USERDIR}/benchmarks/${BENCHMARK_NAME}.yaml"
DATA_DIR="${USERDIR}/data/${BENCHMARK_NAME}"

"${PYTHON_BIN}" scripts/prepare_pmlb_classification_benchmark.py \
  --task-type regression \
  --pmlb-root "${PMLB_ROOT}" \
  --output-benchmark "${BENCHMARK_FILE}" \
  --output-data-dir "${DATA_DIR}" \
  --userdir "${USERDIR}"

"${PYTHON_BIN}" runbenchmark.py "${FRAMEWORK}" "${BENCHMARK_NAME}" "${CONSTRAINT}" \
  -u "${USERDIR}" \
  -o "results_${BENCHMARK_NAME}_${FRAMEWORK}_${CONSTRAINT}" \
  --resume
