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

- **Accelera Pipe graph engine**: build DAG-style ML workflows with
  preprocessing, model, predict, metric, merge, and branch nodes, then reuse the
  selected fitted path through an `ExecutedGraph`.
- **Optimized pipeline execution**: run independent branches in parallel through
  the C++ graph backend, share duplicate first branch nodes, release temporary
  training data after its last consumer, and save/load pipelines as pickle files.
- **Custom model support**: plug in sklearn-compatible estimators or extend
  `CustomClassifier`, `CustomRegressor`, `CustomClusterer`, and
  `CustomTransformer`.
- **Reporting**: generate graph visualizations and HTML metric reports through
  `GraphReport`, `ModelReport`, and autopreprocessing reports.
- **Auto preprocessing**: tabular, text, and image-classification
  preprocessing utilities with saved preprocessors and visual
  summaries.
- **AutoML model selection**: search classification and regression models
  under time and trial budgets, warm-start searches from dataset
  metafeatures, and optionally build voting or stacked ensembles.
- **Dataset retriever**: list and download shared CSV datasets into a local
  cache with `accelera.src.utils.dataset_retriever.DatasetRetriever`.
- **Python/C++ code parallelizer**: convert supported Python loops to C++,
  analyze C/C++ loops with Clang AST, classify safe OpenMP opportunities with
  rules and the OpenMP classifier service, and emit parallelized C++.
- **Accelera Pipe parallelizer integration**: automatically optimize eligible
  custom preprocessing functions or `CustomTransformer` methods, compile them as
  Python-callable native modules, cache compiled outputs, and attach them back to
  graph preprocessing nodes.
- **Benchmark backend prototype**: Express/MongoDB backend scaffolding for
  benchmarks, users, metrics, and submissions.


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
pip install -e .
```

The editable install makes `accelera` importable and keeps it linked to your
checkout while you develop. If you do not install the package, add Accelera to
Python's import path for the current terminal session instead:

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
cmake --build build --parallel
```

When using the `PYTHONPATH` alternative, set it again whenever you open a new
terminal. Imports such as `from accelera.src...` or the native `graph` binding
may otherwise fail even when the package files exist locally.
CMake also checks for Graphviz `dot` and installs it automatically on supported
Windows and Debian/Ubuntu Linux systems so graph-rendering examples can run.

### Node installation

This project requires **Node.js v22.23.0**.

**Linux / macOS**

Using `nvm` is recommended:

```bash
nvm install 22.23.0
nvm use 22.23.0
node -v
npm -v
````

**Windows**

Install **Node.js v22.23.0** from the official Node.js website, then verify the installation:

```powershell
node -v
npm -v
```

The `node -v` command should print:

```bash
v22.23.0
```

### Needed Datasets

To run the AutoPreprocessing demo correctly, you need to download two Kaggle datasets used in this project:

- https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset  
- https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/data  

### Setup Instructions
1. Download both datasets from Kaggle.
2. Extract the downloaded archives.
3. From the first dataset, locate the folder named `PetImages` that is created after extraction and copy it to the `examples/` directory in the project root.
4. From the second dataset, locate the folders named `images` and `masks` that are created after extraction and copy both folders to the `examples/` directory in the project root.
5. After completing these steps, the directory structure should look similar to:
  ``` 
accelera/
├── examples/
│   ├── PetImages/
│   ├── images/
│   └── masks/
└── ...
```
The datasets are required for image classification and segmentation demos.
### Colab Quick Start

If you prefer to run the demo without setting everything up locally, you can use the provided Colab notebook. The notebook demonstrates the full setup flow, including preparing the datasets and running the project Makefile.

Colab notebook: [Demo](https://colab.research.google.com/drive/1J-22PPwm26Hs_OPI_L_ovZlffwl9HP4N?usp=sharing)

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

# Run auto preprocessing demo
python examples/auto_preprocessing_demo.py

# Run AutoML classification and regression examples
python examples/run_classification_task_automl.py
python examples/run_regression_task_automl.py --time-budget 300 --n-trials 10

# Run benchmark backend server
cd accelera/src/benchmark/backend
npm install
npm run dev

# Run benchmark frontend
cd accelera/src/benchmark/frontend
npm install
npm run dev

# Run tests
pytest accelera
```
To run all demo scripts in order, run:

```bash
make -f examples/Makefile
```

On Windows, install `make` first:

```powershell
winget install MSYS2.MSYS2
C:\msys64\usr\bin\bash.exe -lc "pacman -Syu --noconfirm make"
$env:Path = "C:\msys64\usr\bin;$env:Path"
make -f .\examples\Makefile
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

The examples below assume you already ran the Quick Start setup and either
installed the package or exported `PYTHONPATH`.

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
from accelera.src.parallelizer import parallelizer

parallelizer.parallelize("examples/loop_example.py")
# Writes parallelized_loop_example.c in the repo root by default
```

Pass `output_dir="some/path"` if you want the generated file written somewhere
else.

For in-memory C/C++ code:

```python
from accelera.src.parallelizer import parallelizer

code = """
int main() {
    int total = 0;
    for (int i = 0; i < 1000; i++) {
        total += i;
    }
}
"""

parallelized_code = parallelizer.parallelize(code)
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

For internal defined python methods:

```python
import numpy as np

from accelera.src.parallelizer import parallelizer


def normalize_rows(X):
    for i in range(len(X)):
        s = 0
        for j in range(len(X[i])):
            s += X[i][j] * X[i][j]

        norm = s**0.5

        for j in range(len(X[i])):
            X[i][j] = X[i][j] / norm

    return X


results = parallelizer.parallelize(normalize_rows)(
    np.random.rand(500_000, 25).astype(np.float32)
)
print(results)
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
it returns X_train, y_train, X_val, y_val generated from this pipeline and also 

```python
from accelera.src.autopreprocessing.core.classical_training_preprocessing import (
    ClassicalTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

retriever.connect()
df = retriever.retrieve_dataset(dataset_name, url=drive_link, df=True)
X_train, y_train, X_test, y_test =ClassicalTrainingPreprocessing(
  df=df,
  target_col,
  problem_type,
  folder_path,
  val_size,
  random_state,
  cardinality_threshold,
  max_unique_ordinal,
  missing_threshold,
  columns_need_to_drop,
  feature_importance_threshold,
).common_preprocessing()
retriever.close()
```
parameters:
- `df`: Input dataset as a pandas DataFrame.
- `target_col`: Name of the target column.
- `problem_type`: Either `"classification"` or `"regression"`.
- `folder_path`: Directory where generated reports and files will be saved.
- `val_size`: Proportion of the dataset used for validation.
- `random_state`: Random seed for reproducibility.
- `cardinality_threshold`: Threshold used to identify high-cardinality categorical features.
- `max_unique_ordinal`: Maximum number of unique values for ordinal integers feature detection.
- `missing_threshold`: Threshold for dropping columns with missing values.
- `columns_need_to_drop`: List of columns to remove before preprocessing.
- `feature_importance_threshold`: Minimum feature importance score required to keep a feature.
`is_report`: Enable or disable generation of HTML reports and visualizations.
#### Note
You can deal with dataset in these two options
- If you have dataset csv file locally use it 
- You can use your own dataset by providing a Google Drive download link to the retriever.

for the second option :

1. Make sure the file is shared with **Anyone with the link** .
2. Copy the file ID from the Google Drive sharing link.
3. Create a dataset URL using the format:
#### Example
Given the Google Drive link:

```text
https://drive.google.com/file/d/1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs/view?usp=sharing
```

The file ID is:

```text
1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs
```

Create the dataset URL:

```python
drive_link = "https://drive.google.com/uc?id=1VMtLcWDcigwkimpf-eWVMZ7zJMUf7wxs"
```

### Text Auto Preprocessing

Text preprocessing prepares a text column and target column for NLP
experiments. Pass the dataframe, the target column, and the text column, then
use the returned train/validation arrays in your model code.

```python
import pandas as pd

from accelera.src.auto_preprocessing.core.text_training_preprocessing import (
    TextTrainingPreprocessing,
)
from accelera.src.utils.dataset_retriever import retriever

retriever.connect()
df = retriever.retrieve_dataset(dataset_name, url=drive_link, df=True)
TextTrainingPreprocessing(
        df,
        target_col,
        text_col,
        folder_path,
        val_size,
        random_state,
        tfidf_max_features,
        tfidf_ngram,
        tfidf_max_df,
        tfidf_min_df,
        is_report,
)
X_train, y_train, X_val, y_val = text_preprocessor.common_preprocessing()
retriever.close()
```
Parameters

- `df`: Input dataset retrieved from Google Drive.
- `target_col`: Name of the target column.
- `text_col`: Name of the text column used for NLP processing.
- `folder_path`: Directory where reports and outputs are saved.
- `val_size`: Validation split ratio.
- `random_state`: Random seed for reproducibility.
- `tfidf_max_features`: Maximum number of TF-IDF features.
- `tfidf_ngram`: N-gram range for TF-IDF (e.g., (1,2)).
- `tfidf_max_df`: Ignore terms with document frequency higher than this threshold.
- `tfidf_min_df`: Ignore terms with document frequency lower than this threshold.
- `is_report`: Enable or disable HTML report generation.
### Image Auto Preprocessing Classifcation
When `split_training=True`, it creates a validation split from the training
folder. Use `augment=True` when you want training-time augmentation.
Expected Folder Structure

```
Training/
├── Cats/
└── Dogs/

Validation/
├── Cats/
└── Dogs/
```
```python
from accelera.src.auto_preprocessing.core.classification_image_training_preprocessing import (
    ClassificationImageTrainingPreprocessing,
)

image_preprocessor = ClassificationImageTrainingPreprocessing(
    training_folder_images=Training,  # replace with your Training folder
    folder_path=report_folder_path,
    validation_folder_images=Validation, # replace with your Validation folder if exist
    split_training=True,
    val_size=0.2,
    images_size=(224, 224),
    augment=True,
)
training_loader, validation_loader = image_preprocessor.common_preprocessing()
```
### Image Auto Preprocessing Segmentation
This for binary segmentation problem
When `split_training=True`, it creates a validation split from the training
folder. Use `augment=True` when you want training-time augmentation.

Expected Folder Structure
```
Dataset/
├── Training/
│   ├── Images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── img3.jpg
│   │
│   └── Masks/
│       ├── img1.png
│       ├── img2.png
│       └── img3.png
│
└── Validation/
    ├── Images/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── img3.jpg
    │
    └── Masks/
        ├── img1.png
        ├── img2.png
        └── img3.png
```
```python
from accelera.src.auto_preprocessing.core.segmentation_image_training_preprocessing import (
    SegmentationImageTrainingPreprocessing,
)

training_loader, validation_loader = SegmentationImageTrainingPreprocessing(
        training_folder_images=Training_Images,# replace with your Training folder Images
        training_folder_masks=training_folder_masks,# replace with your Training folder mask
        folder_path=folder_path,
        binary_mask_threshold=128,
        validation_folder_images=None,# replace with your Validation folder Images (if exists)
        ,
        validation_folder_masks# replace with your Validation folder masks (if exists)
        augment=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotation=True,
        split_training=True,
        val_size=0.2,
        images_size=image_size,
    ).common_preprocessing()
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

### AutoML Classification and Regression

Accelera AutoML searches across supported scikit-learn, CatBoost, LightGBM,
and XGBoost estimators. Cross-validation ranks candidate configurations within
the supplied time and trial limits. Dataset metafeatures can warm-start the
search, and the best candidates can be combined into voting or stacked
ensembles.

Install Accelera before using the public API:

```bash
pip install .
```

Classification example:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from accelera.src.accelera_automl import AutoMLClassifier

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=12,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = AutoMLClassifier(
    time_budget=300,
    n_trials=20,
    cv=3,
    random_state=42,
    stack_n_jobs=-1,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("accuracy", accuracy_score(y_test, predictions))
print(model.return_leaderboard(top_n=3))
```

For regression, use the same estimator-style interface:

```python
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from accelera.src.accelera_automl import AutoMLRegressor

X, y = make_regression(
    n_samples=1000,
    n_features=20,
    n_informative=12,
    noise=5.0,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = AutoMLRegressor(
    time_budget=300,
    n_trials=20,
    cv=3,
    random_state=42,
    stack_n_jobs=-1,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("r2", r2_score(y_test, predictions))
print(model.return_leaderboard(top_n=3))
```

The repository also contains complete CSV-based examples:

```bash
python examples/run_classification_task_automl.py
python examples/run_regression_task_automl.py --time-budget 300 --n-trials 10
```

These scripts read prepared splits from `data/accelera_automl/`. The regression
script accepts `--time-budget`, `--n-trials`, `--cv`, and
`--stack-n-jobs` options.


### Deployment Module

The deployment module provides a dynamic, procedural version control system (VCS) to track configuration files and multi-stage ML pipelines, enabling automated container builds and deployments to remote target environments like AWS EC2.

#### Dataset and ML Pipeline Details

The demo pipeline uses the classic **Wine Dataset** (`sklearn.datasets.load_wine`) to train and evaluate multiple models:
- **Features**: 13 chemical analyses of wine (alcohol, malic acid, ash, alkalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, od280/od315 of diluted wines, proline).
- **Target**: 3 distinct classes of wine cultivars.

The trained and committed artifacts form a sequenced execution pipeline:
1. **`preprocessing_pipeline`**: Standardizes numerical columns and imputes missing features.
2. **`feature_selector`**: Selects the top $K$ features using univariate feature selection (`SelectKBest`).
3. **`final_model`**: The best performing trained estimator selected automatically (Logistic Regression, Random Forest, or Histogram-Based Gradient Boosting).

#### Prerequisites and Installation

##### 1. Local Environment Setup
To run the VCS commands or execute the end-to-end demo locally, install the dependencies from the root `requirements.txt`:

```bash
# Install local packages for modeling, validation, serving, and deployment
pip install -r requirements.txt
```

These include:
- `fastapi` and `uvicorn` (to host the REST server and UI)
- `great-expectations` (for data schema validation)
- `scikit-learn` & `category-encoders` (for model execution)
- `pydantic`, `pandas`, and `numpy`

##### 2. Remote Host Setup (EC2 / Ubuntu Linux)
- Ensure you have SSH access to your Ubuntu server (e.g., via private key file `.pem`).
- The deployment process will package the pipeline into a Docker container.
- If Docker is not already installed on the target machine, the script will install it automatically when the `--install-docker` flag is provided.

##### 3. Heroku Environment Setup
- Register for an account at [Heroku Sign Up](https://signup.heroku.com/) or log in to [Heroku Dashboard](https://id.heroku.com/).
- Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).
- Before deploying, authenticate your CLI sessions:
  ```bash
  # Log in to your Heroku Account
  heroku login

  # Log in to the Heroku Container Registry
  heroku container:login
  ```
  Alternatively, you can run these logins using the Accelera wrapper CLI:
  ```bash
  python accelera/src/deployment/deployment.py heroku-login
  python accelera/src/deployment/deployment.py heroku-container-login
  ```

#### Running the End-to-End Demo

The repository contains an end-to-end demo script that downloads the SSH key, initializes a registry, trains the pipeline, commits the artifacts, and deploys it to a target EC2 instance:

```bash
# FOR REVIEWRS please use this to run the deployment module easily
python examples/deployment_demo.py
```

#### CLI Deployment Commands

You can run individual deployment commands manually:

```bash
# Initialize VCS registry
python -m accelera.src.deployment.vcs init

# Commit latest models and config
python -m accelera.src.deployment.vcs commit -m "Your commit message"

# View status
python -m accelera.src.deployment.vcs status

# View commit logs
python -m accelera.src.deployment.vcs log

# Deploy the module to an EC2 instance
python accelera/src/deployment/deployment.py ec2-deploy \
  --host <HOST_IP> \
  --user <USER> \
  --key <PATH_TO_PRIVATE_KEY> \
  --install-docker

# Deploy the module to Heroku
python accelera/src/deployment/deployment.py heroku-deploy \
  --app <HEROKU_APP_NAME> \
  --create
```

#### Running from a Custom Directory

If you are running the deployment or VCS commands from a custom directory  you must prefix the commands with `PYTHONPATH` set to the project root `/home/mazen/Desktop/GP/Accelera` so Python can resolve the `accelera` imports. You must ensure that `config.json` and the model artifacts are present for more details see the deployment documentation
  
1. **Initialize the VCS registry** inside your custom directory:
   ```bash
   PYTHONPATH=/home/mazen/Desktop/GP/Accelera python -m accelera.src.deployment.vcs init
   ```

2. **Commit models and configuration** to the local registry:
   ```bash
   PYTHONPATH=/home/mazen/Desktop/GP/Accelera python -m accelera.src.deployment.vcs commit -m "Your commit message"
   ```

3. **Build and Run the Service Locally**:
   ```bash
   PYTHONPATH=/home/mazen/Desktop/GP/Accelera python /home/mazen/Desktop/GP/Accelera/accelera/src/deployment/deployment.py local
   ```

#### Running Deployment Tests
```bash
# Run all deployment module unit tests
PYTHONPATH=accelera/src/deployment:accelera/src pytest accelera/src/deployment/tests
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
│── benchmark/                   # Node.js backend prototype
├── accelera/
│   ├── api/                     # generated public API modules
│   ├── bindings/                # pybind11 bindings
│   └── src/
|       |── accelera_automl/
│       ├── accelera_pipe/       # DAG pipeline, execution graph
│       ├── auto_preprocessing/  # preprocessing, reports 
│       ├── custom/              # estimator base classes
│       ├── deployment/          # deployment module
│       ├── parallelizer/        # parallelizer module
│       ├── utils/               # dataset retriever, parallelizer and code utilities
│       └── wrappers/            # HTML/report helpers
├── src/                         # C++ core, nodes, AST, and utility sources
├── include/                     # C++ headers
├── examples/                    # scripts and notebooks
├── docs/                        # MkDocs documentation
├── shell/                       # setup scripts
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
