<p align="center">
  <img src="docs/accelera.png" alt="Accelera logo" width="180">
</p>

# Accelera

Accelera is a hybrid Python/C++ machine learning framework for building
graph-based pipelines, running independent branches in parallel, generating
HTML reports, and experimenting with automated preprocessing and loop
parallelization.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C++-20-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Features

- **Graph ML pipelines**: build DAG-style workflows with preprocessing, model,
  predict, metric, merge, and branch nodes.
- **Parallel branch execution**: compare multiple preprocessing/model/metric
  combinations in one pipeline run through the C++ graph backend.
- **Custom model support**: plug in sklearn-compatible estimators or extend
  `CustomClassifier`, `CustomRegressor`, `CustomClusterer`, and
  `CustomTransformer`.
- **Reporting**: generate graph visualizations and HTML metric reports through
  `GraphReport`, `ModelReport`, and AutoML preprocessing reports.
- **Auto preprocessing**: tabular, text, image-classification, and
  segmentation preprocessing utilities with saved preprocessors and visual
  summaries.
- **Dataset retriever**: list and download shared CSV datasets into a local
  cache with `accelera.src.utils.dataset_retriever.DatasetRetriever`.
- **C/C++ code parallelizer**: extract loops with Clang AST, derive loop
  features, call an OpenMP classifier service, and inject OpenMP pragmas
  into parallelizable `for` loops.
- **Benchmark backend prototype**: Express/MongoDB backend scaffolding for
  benchmarks, users, metrics, and submissions.

## Current Status

- The core DAG pipeline, custom estimator interfaces, reports, dataset
  retrieval, and preprocessing utilities are implemented in this repo.
- The AutoML search agent API exists, but the default search algorithm is
  still a placeholder.
- The benchmark backend is an early prototype.
- The code parallelizer requires LLVM/Clang, built pybind bindings, and the
  classifier endpoint configured in `accelera/src/config.py`. CMake attempts
  to install LLVM/Clang automatically on Linux and Windows when possible.

## Quick Start

### Installation

```bash
git clone https://github.com/Mohamed-Ashraf273/accelera.git
cd accelera

python -m venv .venv
```

### Activate Virtual Environment

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Add Accelera to Python's import path for this terminal session. This is
required before running examples, notebooks, or tests from the repo.

Linux/macOS:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
```

Windows CMD:

```cmd
set PYTHONPATH=%CD%;%PYTHONPATH%
```

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

Set `PYTHONPATH` again whenever you open a new terminal.
If you skip it, imports such as `from accelera.src...` or the native `graph`
binding may fail even when the package files exist locally.

### Run Examples

```bash
# Accelera Pipe graph pipeline demo
python examples/accpipe_demo.py

# Accelera Pipe with parallelized custom preprocessing
python examples/parallel_accpipe.py

# Code parallelizer benchmark on a hard loop example
python examples/code_parallelizer_demo.py

# Save and load pipeline or executed graph demo
python examples/save_load.py

# Run tests
pytest accelera
```
To run all demo scripts in order, run:

```bash
make -f examples/Makefile
```

## Minimal Usage

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from accelera.src.accelera_pipe.core.pipeline import Pipeline

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    random_state=42,
)
X_test, y_test = X[:200], y[:200]

pipe = Pipeline()
pipe.branch(
    "preprocessing",
    pipe.preprocess("standard", StandardScaler(), branch=True),
    pipe.preprocess("minmax", MinMaxScaler(), branch=True),
).model(
    "logreg",
    LogisticRegression(max_iter=1000),
).predict(
    "predict",
    test_data=X_test,
).metric(
    "accuracy",
    "accuracy_score",
    y_true=y_test,
)

predictions, executed_graph = pipe(X, y, select_strategy="max")
best_result = executed_graph(X_test, y_test)
print(predictions)
print(best_result)
```

## More Usage Examples

The examples below assume you already ran the Quick Start setup and exported
`PYTHONPATH`.

If the native graph import fails, rebuild the C++ bindings with
`cmake --build build -j"$(nproc)"` and run the export command again.

### Accelera Pipe Branch Selection

Use `Pipeline` when you want to compare several preprocessing/model paths in a
single graph run. Each builder call adds a node. Passing `branch=True` creates
a branch candidate, and `branch()` groups those candidates under one split.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from accelera.src.accelera_pipe.core.pipeline import Pipeline

X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_val, y_val = X[:500], y[:500]
X_test, y_test = X[500:1000], y[500:1000]

pipe = Pipeline()
pipe.branch(
    "preprocessing",
    pipe.preprocess("standard", StandardScaler(), branch=True),
    pipe.preprocess("minmax", MinMaxScaler(), branch=True),
).model(
    "model_lr",
    LogisticRegression(max_iter=1000),
).predict(
    "predict",
    test_data=X_val,
).metric(
    "metric",
    "accuracy_score",
    y_true=y_val,
)

results, executed_graph = pipe(X, y, select_strategy="max")
test_results = executed_graph(X_test, y_true=y_test)
print(results)
print(test_results)
```

Useful pipeline options:

- `select_strategy="all"` returns all graph paths.
- `select_strategy="max"` selects the path with the highest metric.
- `select_strategy="min"` selects the path with the lowest metric.
- `custom_strategy=` accepts a user-defined path selection function.
- `pipe.disable_parallel_execution()` forces serial graph execution.
- `pipe.set_multicore_threshold(n)` changes the backend threshold for
  multicore execution.
- `cache=False` is the default for preprocess/model nodes. Enable cache only
  when repeated runs reuse the same expensive node inputs.

### Save and Load Pipelines

Unexecuted pipelines and executed graphs can both be saved as one pickle file.
Save an unexecuted pipeline when you want to store the graph recipe and train
it later. Save an executed graph when you already trained the pipeline and want
to reuse the fitted preprocessing/model path for inference.

```python
from accelera.src.accelera_pipe.core.executed_graph import ExecutedGraph
from accelera.src.accelera_pipe.core.pipeline import Pipeline

pipe.save("pipeline.pkl")
loaded_pipe = Pipeline.load("pipeline.pkl")
results, executed_graph = loaded_pipe(X, y, select_strategy="max")

executed_graph.save("executed_graph.pkl")
loaded_graph = ExecutedGraph.load("executed_graph.pkl")
predictions = loaded_graph(X_test, y_true=y_test)
```

Notes:

- An unexecuted pipeline stores the graph structure and callable node objects.
- An executed graph stores the fitted objects needed for inference.
- Top-level custom preprocess functions and simple lambdas can be stored
  through source-backed wrappers when possible.
- Custom functions with closures are rejected because captured external
  variables cannot be reconstructed safely from source code alone.


### C/C++ Loop Parallelization

Use the parallelizer when you want to analyze loop-heavy C/C++ code and emit
OpenMP pragmas. The module needs the C++ bindings, LLVM/Clang, and the
classifier endpoint configured in `accelera/src/config.py`.

```python
from accelera.src.utils.parallelizer import parallelizer

parallelizer.parallelize("examples/loop_example.py")
# Writes parallelized_test_loops.c in the repo root by default
```

Pass `output_dir="some/path"` if you want the generated file written somewhere
else.

For in-memory C/C++ code:

```python
from accelera.src.utils.parallelizer import parallelizer

code = """
int main() {
    int total = 0;
    for (int i = 0; i < 1000; i++) {
        total += i;
    }
}
"""

parallelized_code = parallelizer.parallelize(code, file=False)
print(parallelized_code)
```

For supported Python code, the parallelizer first converts Python to C++ and
then applies the same loop extraction and OpenMP insertion path:

```python
code = """
total = 0
for i in range(1000):
    total += i
print(total)
"""

parallelized_code = parallelizer.parallelize(code)
print(parallelized_code)
```

The Python-to-C++ converter supports a restricted loop-friendly subset:

- constants, variables, arithmetic, comparisons, boolean operations;
- function calls and `print`;
- attribute access and indexing, but not slices;
- simple assignment and simple-name augmented assignment;
- `if`/`else`, `return`;
- `for i in range(...)` with one, two, or three arguments;
- simple `def` functions without decorators.

Unsupported Python syntax raises an error or falls back to the original Python
function when used through automatic pipeline optimization.

### Automatic Custom Preprocessing Acceleration in Accelera Pipe

`Pipeline.preprocess()` automatically tries to optimize custom preprocessing
functions through the Parallelizer when possible.

```python
from accelera.src.accelera_pipe.core.pipeline import Pipeline

def normalize_rows(X):
    for i in range(len(X)):
        s = 0
        for j in range(len(X[i])):
            s += X[i][j] * X[i][j]
        norm = s ** 0.5
        for j in range(len(X[i])):
            X[i][j] = X[i][j] / norm
    return X

pipe = Pipeline()
pipe.preprocess("normalize", normalize_rows)
```

The automatic path is:

```text
Python custom function
-> py2cpp_converter
-> parallelizer OpenMP pragma insertion
-> cpp_compiler.py / pybind11 native module
-> Accelera Pipe preprocess node
```

If conversion, classification, OpenMP insertion, compilation, or import fails,
Accelera keeps the original Python function so the pipeline remains correct.

### Dataset Retriever

Use the dataset retriever when you want to pull one of the shared demo datasets
without manually downloading CSV files. Call `available_datasets()` first to
see the registered names, then connect, retrieve the dataset, and close the
connection when finished.

```python
from accelera.src.utils.dataset_retriever import retriever

print(retriever.available_datasets())

retriever.connect()
housing_df = retriever.retrieve_dataset("Housing", df=True)
print(housing_df.head())
retriever.close()
```

### Tabular Auto Preprocessing

Tabular preprocessing prepares classical machine-learning datasets. It handles
common cleaning, train/validation splitting, target handling, and report output
under the folder you pass in `folder_path`.

```python
from accelera.src.automl.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

retriever.connect()
df = retriever.retrieve_dataset("Titanic-Dataset", df=True)

preprocessor = ClassicalTrainingPreprocessing(
    df,
    target_col="Survived",
    problem_type="classification",
    folder_path="./titanic_preprocessing_report",
)
X_train, y_train, X_val, y_val = preprocessor.common_preprocessing()

retriever.close()
```

### Text Auto Preprocessing

Text preprocessing prepares a text column and target column for NLP
experiments. Pass the dataframe, the target column, and the text column, then
use the returned train/validation arrays in your model code.

```python
import pandas as pd

from accelera.src.automl.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)

reviews_df = pd.DataFrame(
    {
        "review": ["Great product", "Very bad experience", "I like it"],
        "class": [1, 0, 1],
    }
)

text_preprocessor = TextTrainingPreprocessing(
    reviews_df,
    target_col="class",
    text_col="review",
    folder_path="./reviews_report",
)
X_train, y_train, X_val, y_val = text_preprocessor.common_preprocessing()
```

### Image Auto Preprocessing

Image preprocessing expects a folder structure that contains class folders.
When `split_training=True`, it creates a validation split from the training
folder. Use `augment=True` when you want training-time augmentation.

```python
from accelera.src.automl.core.classification_image_training_preprocessing import (
    ClassificationImageTrainingPreprocessing,
)

image_preprocessor = ClassificationImageTrainingPreprocessing(
    training_folder_images="./PetImages",  # replace with your class folders
    folder_path="./PetImagesReport",
    split_training=True,
    val_size=0.2,
    images_size=(224, 224),
    augment=True,
)
training_loader, validation_loader = image_preprocessor.common_preprocessing()
```

### Pipeline Graph Report

Graph reports visualize a serialized pipeline graph together with the pipeline
results. Serialize the pipeline to XML first, then pass that XML file and the
results to `GraphReport`.

```python
from accelera.src.utils.accelera_utils import serialize
from accelera.src.accelera_pipe.wrappers.graph_report import GraphReport

predictions, executed_graph = pipe(X, y, select_strategy="max")
serialize(pipe, "pipeline.xml")

report = GraphReport("pipeline_report", "pipeline.xml", predictions)
report.execute()
```

### Standalone Model Report

Use `ModelReport` when you already have metric results from a normal model and
want the same report format without building a full Accelera Pipe graph.

```python
from sklearn.metrics import accuracy_score

from accelera.src.accelera_pipe.wrappers.model_report import ModelReport

accuracy = accuracy_score(y_test, model.predict(X_test))
results = [
    {
        "metric name": "accuracy",
        "result": accuracy,
        "plot_func": None,
        "labels_name": None,
        "headers_name": None,
    }
]

report = ModelReport("model_report", results=results)
report.execute()
```

### Runtime Requirements and Common Blockers

Things that can prevent these modules from running:

- **Missing C++ bindings**: run `cmake --build build`.
- **LLVM/Clang missing**: on Debian/Ubuntu Linux, CMake attempts to install it
  automatically when it can use root or non-interactive sudo. On Windows,
  CMake attempts to install LLVM automatically with `winget` first, then
  Chocolatey if available.
- **OpenMP compiler support missing**: generated native code requires a compiler
  with OpenMP support. On Linux this usually means `g++`/`clang++` plus OpenMP
  runtime libraries. On Windows, use a compiler/toolchain with OpenMP enabled.
- **Classifier endpoint unavailable**: set
  `ACCELERA_CLASSIFIER_ENDPOINT` or ensure the default Hugging Face Space is
  reachable. Also check `ACCELERA_REQUEST_TIMEOUT_S` for slow networks.
- **`clang-format` missing**: output formatting is optional. Install
  `clang-format` or set `ACCELERA_ENABLE_CPP_FORMATTING=0`.
- **Unsupported Python syntax**: the converter is intentionally limited. Use
  simple numeric loops, `range`, scalar variables, and indexing.
- **Custom function source unavailable**: functions defined dynamically,
  interactively, or inside closures may not be inspectable or saveable.
- **Closure variables in saved custom functions**: source-backed save/load
  rejects closures because external captured values are not stored.
- **Pickle limitations**: custom classes/functions must be pickle-compatible
  unless wrapped by the source-backed function path.
- **Large memory usage in branch-heavy searches**: graph execution may use more
  memory than sklearn Pipeline because multiple branches and fitted states can
  be alive during selection.
- **Cache confusion**: cache is off by default. Enable it only when repeated
  identical node inputs justify the hashing and disk I/O cost.

Useful environment variables:

```bash
export ACCELERA_CLASSIFIER_ENDPOINT="https://accelera-ai-open-mp-classifier.hf.space/predict"
export ACCELERA_REQUEST_TIMEOUT_S=10
export ACCELERA_ENABLE_CPP_FORMATTING=0  # optional
export ACCELERA_CPP_OPT_LEVEL=-O0        # faster compile, default in cpp_compiler
```

Useful validation commands:

```bash
python examples/sklearn_comp.py
python examples/parallel_accpipe.py
python tools/evaluate_hard_parallelizer.py
pytest accelera/src/accelera_pipe/core/pipeline_test.py -q
pytest accelera/src/utils/parallelizer_test.py -q
```

## Project Map

```text
accelera/
├── accelera/
│   ├── api/                 # generated public API modules
│   ├── bindings/            # pybind11 bindings
│   └── src/
│       ├── accelera_pipe/   # DAG pipeline, execution graph
│       ├── automl/          # preprocessing, reports, AutoML agent scaffold
│       ├── benchmark/       # Node.js backend prototype
│       ├── custom/          # estimator base classes
│       ├── utils/           # dataset retriever, parallelizer and code utilities
│       └── wrappers/        # HTML/report helpers
├── src/                     # C++ core, nodes, AST, and utility sources
├── include/                 # C++ headers
├── examples/                # scripts and notebooks
├── docs/                    # MkDocs documentation
├── shell/                   # setup scripts
└── CMakeLists.txt
```

## Useful Commands

```bash
# Regenerate API exports after changing Python modules
python api_gen.py

# Run formatting/lint hooks
pre-commit run --all-files --hook-stage manual

# Serve docs locally
mkdocs serve
# Run Benchmark
## Run Backend
cd accelera/src/benchmark/backend
npm install
npm run dev
## Run Frontend
cd accelera/src/benchmark/frontend
npm install
npm run dev


```

## License

Apache License 2.0. See [LICENSE](LICENSE).
