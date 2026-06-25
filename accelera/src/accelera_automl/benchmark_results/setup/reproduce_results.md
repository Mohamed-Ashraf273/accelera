# Reproduce The AutoMLBenchmark Classification And PMLB Regression Benchmarks

## Repository Layout

Keep these repositories as siblings under the same parent directory:

```text
WORKDIR/
  accelera_automl/
  automlbenchmark/
  pmlb/
```

This is important because the custom AMLB adapter inserts `WORKDIR/` into `sys.path` and then imports `accelera_automl` directly from there.

## Files That Must Exist In `accelera_automl`

The custom AMLB overlay lives in:

```text
accelera_automl/automlbenchmark_custom_myautoml/
```

Required files:

```text
automlbenchmark_custom_myautoml/config.yaml
automlbenchmark_custom_myautoml/constraints.yaml
automlbenchmark_custom_myautoml/frameworks.yaml
automlbenchmark_custom_myautoml/extensions/MyAutoML/__init__.py
automlbenchmark_custom_myautoml/extensions/MyAutoML/exec.py
automlbenchmark_custom_myautoml/scripts/prepare_pmlb_classification_benchmark.py
```

Useful benchmark files:

```text
automlbenchmark_custom_myautoml/benchmarks/flaml_paper_39.yaml
automlbenchmark_custom_myautoml/benchmarks/pmlb_regression.yaml
```

Useful helper scripts:

```text
automlbenchmark_custom_myautoml/run_automlbenchmark_classification_framework.sh
automlbenchmark_custom_myautoml/run_automlbenchmark_classification_frameworks.sh
automlbenchmark_custom_myautoml/run_pmlb_regression_framework.sh
```

Notes:

- `frameworks.yaml` in this folder defines only `accelera_automl`.
- `flaml`, `autosklearn`, and `TPOT` come from upstream `automlbenchmark`.
- `prepare_pmlb_classification_benchmark.py` is required for the PMLB regression benchmark because it is not part of a clean upstream AMLB clone in this setup.

## Benchmark Versions Used In The Recorded Setup

For closest reproduction, use:

- `automlbenchmark` commit: `73823c55d8674761aab25520a75cbeea065ca124`
- Python: `3.8.10`
- Constraint: `pmlb_30m`
- Per-dataset budget: `1800` seconds
- Folds: `1`
- Cores: `1`

Key package versions from the recorded AMLB environment:

- `openml==0.13.1`
- `numpy==1.24.2`
- `pandas==1.5.3`
- `scikit-learn==1.2.2`
- `scipy==1.10.1`
- `ConfigSpace==1.2.0`
- `lightgbm==4.6.0`
- `xgboost==2.1.4`
- `catboost==1.2.10`

## 1. Clone The Repositories

```bash
mkdir -p ~/work
cd ~/work

git clone https://github.com/openml/automlbenchmark.git
git clone https://github.com/EpistasisLab/pmlb.git
```

Check out the AMLB revision used in the recorded setup:

```bash
cd ~/work/automlbenchmark
git checkout 73823c55d8674761aab25520a75cbeea065ca124
```

If your `pmlb` checkout uses Git LFS, make sure the real dataset files are present.

## 2. Create The AMLB Virtual Environment

For closest reproduction:

```bash
cd ~/work/automlbenchmark
python3.8 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

All commands below use:

```bash
cd ~/work/automlbenchmark
AMLB_PY=./venv/bin/python
```

If `python3.8` is not available, use the closest Python version you can, but exact reproducibility may change.

## 3. Copy The Custom Overlay Into The AMLB Clone

Create the AMLB target directories:

```bash
cd ~/work/automlbenchmark
mkdir -p examples/custom_myautoml/extensions/MyAutoML
mkdir -p examples/custom_myautoml/benchmarks
mkdir -p scripts
```

Copy the required overlay files:

```bash
cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/config.yaml \
  examples/custom_myautoml/config.yaml

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/constraints.yaml \
  examples/custom_myautoml/constraints.yaml

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/frameworks.yaml \
  examples/custom_myautoml/frameworks.yaml

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/extensions/MyAutoML/__init__.py \
  examples/custom_myautoml/extensions/MyAutoML/__init__.py

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/extensions/MyAutoML/exec.py \
  examples/custom_myautoml/extensions/MyAutoML/exec.py

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/scripts/prepare_pmlb_classification_benchmark.py \
  scripts/prepare_pmlb_classification_benchmark.py
```

Optional benchmark files:

```bash
cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/benchmarks/flaml_paper_39.yaml \
  examples/custom_myautoml/benchmarks/flaml_paper_39.yaml

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/benchmarks/pmlb_regression.yaml \
  examples/custom_myautoml/benchmarks/pmlb_regression.yaml
```

Optional helper scripts:

```bash
cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/run_automlbenchmark_classification_framework.sh \
  run_automlbenchmark_classification_framework.sh

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/run_pmlb_regression_framework.sh \
  run_pmlb_regression_framework.sh

cp ~/work/accelera_automl/automlbenchmark_custom_myautoml/run_automlbenchmark_classification_frameworks.sh \
  run_automlbenchmark_classification_frameworks.sh

chmod +x run_automlbenchmark_classification_framework.sh
chmod +x run_automlbenchmark_classification_frameworks.sh
chmod +x run_pmlb_regression_framework.sh
```

## 4. Classification Benchmark

The classification benchmark is the AutoMLBenchmark/OpenML suite stored in:

```text
examples/custom_myautoml/benchmarks/flaml_paper_39.yaml
```

It does not need the local PMLB preparation script.

## 5. Run The Four Libraries On Classification

All runs below use:

- benchmark: `flaml_paper_39`
- constraint: `flaml_paper_1h`
- user directory: `examples/custom_myautoml`

### `accelera_automl`

```bash
cd ~/work/automlbenchmark
AMLB_PY=./venv/bin/python

$AMLB_PY runbenchmark.py accelera_automl flaml_paper_39 flaml_paper_1h \
  -u examples/custom_myautoml \
  -o results_flaml_paper_39_accelera_automl_flaml_paper_1h \
  --resume
```

### `flaml`

```bash
$AMLB_PY runbenchmark.py flaml flaml_paper_39 flaml_paper_1h \
  -u examples/custom_myautoml \
  -o results_flaml_paper_39_flaml_flaml_paper_1h \
  --resume
```

### `autosklearn`

```bash
$AMLB_PY runbenchmark.py autosklearn flaml_paper_39 flaml_paper_1h \
  -u examples/custom_myautoml \
  -o results_flaml_paper_39_autosklearn_flaml_paper_1h \
  --resume
```

### `TPOT`

```bash
$AMLB_PY runbenchmark.py TPOT flaml_paper_39 flaml_paper_1h \
  -u examples/custom_myautoml \
  -o results_flaml_paper_39_TPOT_flaml_paper_1h \
  --resume
```

## 6. Prepare The Local PMLB Regression Benchmark

```bash
cd ~/work/automlbenchmark
AMLB_PY=./venv/bin/python

$AMLB_PY scripts/prepare_pmlb_classification_benchmark.py \
  --task-type regression \
  --pmlb-root ~/work/pmlb \
  --output-benchmark examples/custom_myautoml/benchmarks/pmlb_regression.yaml \
  --output-data-dir examples/custom_myautoml/data/pmlb_regression \
  --userdir examples/custom_myautoml
```

## 7. Run The Four Libraries On Regression

All runs below use:

- benchmark: `pmlb_regression`
- constraint: `pmlb_30m`
- user directory: `examples/custom_myautoml`

### `accelera_automl`

```bash
cd ~/work/automlbenchmark
AMLB_PY=./venv/bin/python

$AMLB_PY runbenchmark.py accelera_automl pmlb_regression pmlb_30m \
  -u examples/custom_myautoml \
  -o results_pmlb_regression_accelera_automl_pmlb_30m \
  --resume
```

### `flaml`

```bash
$AMLB_PY runbenchmark.py flaml pmlb_regression pmlb_30m \
  -u examples/custom_myautoml \
  -o results_pmlb_regression_flaml_pmlb_30m \
  --resume
```

### `autosklearn`

```bash
$AMLB_PY runbenchmark.py autosklearn pmlb_regression pmlb_30m \
  -u examples/custom_myautoml \
  -o results_pmlb_regression_autosklearn_pmlb_30m \
  --resume
```

### `TPOT`

```bash
$AMLB_PY runbenchmark.py TPOT pmlb_regression pmlb_30m \
  -u examples/custom_myautoml \
  -o results_pmlb_regression_TPOT_pmlb_30m \
  --resume
```

## 8. Optional Helper Scripts

Classification, one framework at a time:

```bash
cd ~/work/automlbenchmark
PYTHON_BIN=./venv/bin/python ./run_automlbenchmark_classification_framework.sh accelera_automl flaml_paper_30m flaml_paper_39
PYTHON_BIN=./venv/bin/python ./run_automlbenchmark_classification_framework.sh flaml flaml_paper_30m flaml_paper_39
PYTHON_BIN=./venv/bin/python ./run_automlbenchmark_classification_framework.sh autosklearn flaml_paper_30m flaml_paper_39
PYTHON_BIN=./venv/bin/python ./run_automlbenchmark_classification_framework.sh TPOT flaml_paper_30m flaml_paper_39
```

Regression, one framework at a time:

```bash
cd ~/work/automlbenchmark
PYTHON_BIN=./venv/bin/python ./run_pmlb_regression_framework.sh accelera_automl ~/work/pmlb pmlb_30m
PYTHON_BIN=./venv/bin/python ./run_pmlb_regression_framework.sh flaml ~/work/pmlb pmlb_30m
PYTHON_BIN=./venv/bin/python ./run_pmlb_regression_framework.sh autosklearn ~/work/pmlb pmlb_30m
PYTHON_BIN=./venv/bin/python ./run_pmlb_regression_framework.sh TPOT ~/work/pmlb pmlb_30m
```

Classification, all four in sequence:

```bash
cd ~/work/automlbenchmark
PYTHON_BIN=./venv/bin/python ./run_automlbenchmark_classification_frameworks.sh flaml_paper_30m flaml_paper_39
```

## 9. Where Results Are Written

Classification result directories:

- `results_flaml_paper_39_accelera_automl_flaml_paper_30m`
- `results_flaml_paper_39_flaml_flaml_paper_30m`
- `results_flaml_paper_39_autosklearn_flaml_paper_30m`
- `results_flaml_paper_39_TPOT_flaml_paper_30m`

Regression result directories:

- `results_pmlb_regression_accelera_automl_pmlb_30m`
- `results_pmlb_regression_flaml_pmlb_30m`
- `results_pmlb_regression_autosklearn_pmlb_30m`
- `results_pmlb_regression_TPOT_pmlb_30m`

AMLB may also update its global `results.csv`.

## 10. Important Reproducibility Notes

- The GitHub repo will not ship with a prebuilt virtual environment. Create `./venv/` yourself as shown above.
- Keep `accelera_automl`, `automlbenchmark`, and `pmlb` as sibling directories.
- Use the exact AMLB commit recorded above for the closest match.
- Rebuild the local PMLB benchmark files before running the experiments.
- The commands above use `--resume` because that was part of the recorded workflow.
