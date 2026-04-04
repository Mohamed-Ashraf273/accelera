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
  features, call classifier/generator services, and inject OpenMP pragmas
  into parallelizable `for` loops. This module is Linux-only.
- **Benchmark backend prototype**: Express/MongoDB backend scaffolding for
  benchmarks, users, metrics, and submissions.

## Current Status

- The core DAG pipeline, custom estimator interfaces, reports, dataset
  retrieval, and preprocessing utilities are implemented in this repo.
- The AutoML search agent API exists, but the default search algorithm is
  still a placeholder.
- The benchmark backend is an early prototype.
- The code parallelizer requires Linux, LLVM/Clang, built pybind bindings,
  and the remote classifier/generator endpoints configured in
  `accelera/src/config.py`.

## Quick Start

```bash
git clone https://github.com/Mohamed-Ashraf273/accelera.git
cd accelera

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install psutil requests gdown graphviz

export PYTHONPATH="$PWD"

# Linux only, required before CMake if you want to build code-parallelizer
# bindings and also because the current Linux CMake config expects LLVM.
sudo bash shell/install_llvm.sh 18

cmake -S . -B build
cmake --build build -j"$(nproc)"
```

### Run Examples

```bash
# Parallel sklearn-vs-Accelera pipeline comparison
python examples/sklearn_comp.py

# Full branching pipeline demo with a custom PyTorch classifier and reports
python examples/demo.py

# Run tests
pytest accelera
```

For notebooks, open `examples/dataset_retriever_demo.ipynb`,
`examples/code_optimizer_demo.ipynb`,
`examples/autopreprocessing-classification-v3.ipynb`, or
`examples/segmentation-training-gp.ipynb` after exporting `PYTHONPATH`.

## Minimal Usage

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from accelera.src.core.pipeline import Pipeline

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

### Dataset Retriever

```python
from accelera.src.utils.dataset_retriever import retriever

print(retriever.available_datasets())

retriever.connect()
housing_df = retriever.retrieve_dataset("Housing", df=True)
print(housing_df.head())
retriever.close()
```

### Tabular Auto Preprocessing

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

```python
from accelera.src.utils.accelera_utils import serialize
from accelera.src.wrappers.graph_report import GraphReport

predictions, executed_graph = pipe(X, y, select_strategy="max")
serialize(pipe, "pipeline.xml")

report = GraphReport("pipeline_report", "pipeline.xml", predictions)
report.execute()
```

### Standalone Model Report

```python
from sklearn.metrics import accuracy_score

from accelera.src.wrappers.model_report import ModelReport

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

### C/C++ Loop Parallelization

```python
from accelera.src.core.parallelizer import parallelizer

parallelizer.parallelize("examples/test_loops.c")
# Writes examples/parallelized_test_loops.c
```

## Project Map

```text
accelera/
├── accelera/
│   ├── api/                 # generated public API modules
│   ├── bindings/            # pybind11 bindings
│   └── src/
│       ├── core/            # DAG pipeline, execution graph, parallelizer
│       ├── automl/          # preprocessing, reports, AutoML agent scaffold
│       ├── benchmark/       # Node.js backend prototype
│       ├── custom/          # estimator base classes
│       ├── utils/           # dataset retriever and code utilities
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
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
