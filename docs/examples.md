# Examples

Practical examples demonstrating Accelera's available features. All examples are available in the `examples/` folder.

## Table of Contents

- [Core Pipeline Examples](#core-pipeline-examples)
- [Code Parallelizer Examples](#code-parallelizer-examples)
- [Features Under Implementation](#features-under-implementation)

## Core Pipeline Examples

The core graph-based pipeline module for building ML workflows using Python's Pipeline API.

### Complete Demo with Custom Models

**File**: `examples/demo.py`

Full-featured example showing branching pipelines, custom models, and visualization:

```python
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from accelera.src.core.pipeline import Pipeline
from accelera.src.custom.classifier import CustomClassifier
from accelera.src.wrappers.graph_report import GraphReport

# Define custom PyTorch model
class TorchDenseModel(CustomClassifier):
    def __init__(self, input_dim=None, hidden_dim=32, lr=0.01, 
                 epochs=100, random_state=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def _build_model(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        ).to(self.device)
    
    def fit(self, X, y):
        # Training implementation
        pass
    
    def predict(self, X):
        # Prediction implementation
        pass

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)

# Create branching pipeline to compare multiple models
pipeline = Pipeline()

pipeline.branch(
    "preprocessing",
    pipeline.preprocess("scaler", StandardScaler(), branch=True),
).branch(
    "models",
    pipeline.model("rf", RandomForestClassifier(n_estimators=100), branch=True),
    pipeline.model("lr", LogisticRegression(max_iter=1000), branch=True),
    pipeline.model("svm", SVC(kernel='rbf'), branch=True),
    pipeline.model("torch", TorchDenseModel(hidden_dim=64), branch=True),
).metric("accuracy", "accuracy_score")

# Execute and get results
predictions, executed_graph = pipeline(X, y)

# Generate report
report = GraphReport("ml_pipeline_report", executed_graph)
report.execute()
```

**Run**: `python examples/demo.py`

---

### Sklearn Comparison - Parallel vs Sequential

**File**: `examples/sklearn_comp.py`

Compare Accelera's parallel execution with sklearn's sequential pipeline:

```python
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as skpipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from accelera.src.core.pipeline import Pipeline as accpipeline

# Generate large dataset
X, y = make_classification(
    n_samples=10_000,
    n_features=25,
    n_classes=4,
    n_informative=20,
    random_state=42
)

test_data = X[:500]
y_test = y[:500]

# Accelera branching pipeline - runs in parallel
acc_pipe = accpipeline()
acc_pipe.branch(
    "preprocessing",
    acc_pipe.preprocess("scaler", StandardScaler(), branch=True),
    acc_pipe.preprocess("scaler", MinMaxScaler(), branch=True),
).branch(
    "models",
    acc_pipe.model("model_lr", LogisticRegression(max_iter=1000), branch=True),
    acc_pipe.model("model_svc", SVC(C=10), branch=True),
).predict("predict", test_data=test_data).metric(
    "metric", "accuracy_score", y_true=y_test
)

# Execute - runs 4 pipelines in parallel:
# 1. StandardScaler -> LogisticRegression
# 2. MinMaxScaler -> LogisticRegression
# 3. StandardScaler -> SVC
# 4. MinMaxScaler -> SVC

start = time.time()
predictions, executed_graph = acc_pipe(X, y)
acc_time = time.time() - start

print(f"Accelera parallel execution: {acc_time:.2f}s")
print(f"Total pipelines executed: 4")
```

**Run**: `python examples/sklearn_comp.py`

---

### Classification with Model Reports

**File**: `examples/report_model.py`

Complete classification workflow with Titanic dataset and model reporting:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from accelera.src.wrappers.model_report import ModelReport

# Load and preprocess Titanic dataset
data = pd.read_csv("examples/Titanic-Dataset.csv")
data.drop(
    axis=1, 
    columns=["PassengerId", "Name", "Ticket", "Cabin"], 
    inplace=True
)

# Handle missing values
data["Age"].fillna(data["Age"].mean(), inplace=True)
data.dropna(inplace=True)

# Encode categorical features
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])

# Split data
X = data.drop(columns=["Survived"], axis=1)
y = data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)

# Generate report
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

print(f"Classification Accuracy: {accuracy:.4f}")
```

**Run**: `python examples/report_model.py`

**Datasets**: Uses `examples/Titanic-Dataset.csv` and `examples/Housing.csv`

---

## Code Parallelizer Examples

Extract and analyze loops from C/C++ code (Linux only).

### Loop Extraction

**File**: `examples/extract_loops.py`

Extract loops from C code and save to JSON:

```python
from accelera.src.code_parallelizer.utils.code_utils import extract_loops
from accelera.src.code_parallelizer.utils.code_utils import write_loops_to_json

cpp_file = "examples/test_loops.c"
output_json = "loops_output.json"

print(f"Extracting loops from: {cpp_file}")
print("-" * 60)

# Extract loops
loops = extract_loops(cpp_file)

print(f"\n✓ Found {len(loops)} loops:\n")

# Display each loop
for i, loop in enumerate(loops, 1):
    print(f"Loop {i}:")
    print(f"  Type: {loop.type}")
    print(f"  Lines: {loop.start_line}-{loop.end_line}")
    print("  Code preview:")
    
    # Show first few lines
    code_lines = loop.code.split("\n")
    preview_lines = min(5, len(code_lines))
    for line in code_lines[:preview_lines]:
        print(f"    {line}")
    
    if len(code_lines) > preview_lines:
        print(f"    ... ({len(code_lines) - preview_lines} more lines)")
    print()

# Write to JSON file
if write_loops_to_json(loops, output_json):
    print(f"✓ Successfully written loops to: {output_json}")
    print("\nView with:")
    print(f"  cat {output_json}")
    print(f"  python -m json.tool {output_json}")
else:
    print(f"✗ Failed to write JSON output")
```

**Run**: `python examples/extract_loops.py`

**Input File**: `examples/test_loops.c` (sample C code with various loop types)

---

### Code Optimizer Demo

**File**: `examples/code_optimizer_demo.py`

Convert Python code to optimized C++ with parallelization:

```python
from accelera.src.code_parallelizer import parallelize_code

python_code = """
def LogisticRegression(X, y, learning_rate=0.01, num_iterations=1000):
    import numpy as np
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(num_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = 1 / (1 + np.exp(-linear_model))
        dw = (1 / m) * np.dot(X.T, (y_predicted - y))
        db = (1 / m) * np.sum(y_predicted - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias
"""

# Convert and parallelize
cpp_code = parallelize_code(python_code, "converted_code.cpp")

print("Python code converted to optimized C++")
print(f"Output saved to: converted_code.cpp")
```

**Run**: `python examples/code_optimizer_demo.py`

**Note**: This feature converts Python ML code to parallelized C++ for performance optimization.

---

## Features Under Implementation

The following features are currently being developed and examples will be added when available.

### AutoML Module (Coming Soon)

**File**: `examples/automl_demo.py` (Preview - Not fully functional)

Automatically discover optimal pipelines for your datasets:

```python
import os
import pandas as pd

from accelera.src.automl.core.agent import AutoAccelera

# AutoML agent for automatic pipeline generation
agent = AutoAccelera()

# Load dataset
current_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(current_dir, "Titanic-Dataset.csv"))

# Automatically find best pipeline
best_pipeline = agent.get_pipeline(df, "Survived")

print("Best pipeline discovered!")
```

**Status**: Under implementation - API may change

---

### Deployment Module (Coming Soon)

Deploy and track ML models in production.

```python
# Example API (not yet available)
from accelera.src.deployment import ModelDeployer

deployer = ModelDeployer()
deployer.deploy(model, name="production_model", version="1.0.0")
```

### Benchmark Platform (Coming Soon)

Compete with the community to find the best ML pipelines.

```python
# Example API (not yet available)
from accelera.src.benchmark import BenchmarkSubmission

submission = BenchmarkSubmission(
    dataset_id="titanic",
    pipeline=best_pipeline,
    metrics={"accuracy": 0.87}
)
submission.submit()
```

## Next Steps

- Explore [Core Pipeline API](core-pipeline.md#api-reference) for complete documentation
- Learn about [Code Parallelizer](code-parallelizer.md#api-reference) features
- See [Contributing](contributing.md) to help implement new features

A simple classification pipeline with preprocessing and model training.

```python
from accelera.src.core.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_test, y_test = make_classification(n_samples=200, n_features=20, random_state=43)

# Create pipeline
pipeline = Pipeline()

# Add preprocessing
pipeline.preprocess("scaler", StandardScaler())

# Add model
pipeline.model("rf", RandomForestClassifier(n_estimators=100))

# Add prediction
pipeline.predict("predict", X_test)

# Add metrics
pipeline.metric("accuracy", "accuracy_score", y_true=y_test)

# Execute pipeline
predictions, executed_graph = pipeline(X, y)

print(f"Accuracy: {predictions[0]}")
```

## AutoML Example

Automatically find the best pipeline for the Titanic dataset.

```python
import os
import pandas as pd
from accelera.src.automl.core.agent import AutoAccelera

# Load data
current_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(current_dir, "Titanic-Dataset.csv"))

# Create AutoML agent
agent = AutoAccelera()

# Get best pipeline
best_pipeline = agent.get_pipeline(df, "Survived")

# Use the pipeline
predictions, executed_graph = best_pipeline(X_test, y_test)
```

## Branching Pipeline

Compare multiple preprocessing methods and models in parallel.

```python
import numpy as np
from accelera.src.core.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
p = Pipeline()

# Branch preprocessing
p.branch(
    "preprocessing",
    [
        p.preprocess("standard_scaler", StandardScaler(), branch=True),
        p.preprocess(
            "power_transform",
            lambda x: np.sign(x) * np.power(np.abs(x), 0.8),
            branch=True
        ),
    ],
    p.preprocess(
        "normalize",
        lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8),
        branch=True
    )
)

# Common preprocessing
p.preprocess("clip", lambda x: np.clip(x, -5, 5))

# Branch models
p.branch(
    "models",
    p.model("logreg", LogisticRegression(), branch=True),
    p.model("rf", RandomForestClassifier(), branch=True)
)

# Predict
p.predict("predict", test_data)

# Branch metrics
p.branch(
    "metric",
    p.metric("accuracy", "accuracy_score", y_true=y_test, branch=True),
    p.metric("precision", "precision_score", y_true=y_test, branch=True)
)

# Execute and compare all branches
predictions, executed_graph = p(X, y)
```

## Code Parallelizer

Extract loops from C/C++ code and convert Python to C++.

### Loop Extraction

```python
from code_parallelizer_utils import extract_loops, write_loops_to_json

# Extract loops from C++ file
cpp_file = "examples/test_loops.cpp"
loops = extract_loops(cpp_file, ["-std=c++17"])

print(f"Found {len(loops)} loops")

for i, loop in enumerate(loops, 1):
    print(f"\nLoop {i}:")
    print(f"  Type: {loop.type}")
    print(f"  Lines: {loop.start_line}-{loop.end_line}")
    print(f"  Code:\n{loop.code[:100]}...")

# Save to JSON
write_loops_to_json(loops, "loops_output.json")
```

### Python to C++ Conversion

```python
from accelera.src.code_parallelizer import parallelize_code

python_code = """
def LogesticRegression(X, y, learning_rate=0.01, num_iterations=1000):
    import numpy as np
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(num_iterations):
        linear_model = np.dot(X, weights) + bias
        y_predicted = 1 / (1 + np.exp(-linear_model))
        dw = (1 / m) * np.dot(X.T, (y_predicted - y))
        db = (1 / m) * np.sum(y_predicted - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias
"""

# Convert to optimized C++
cpp_code = parallelize_code(python_code, "converted_code.cpp")
print("C++ code generated successfully")
```

## Custom Models

Implement custom models using PyTorch.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from accelera.src.custom.classifier import CustomClassifier

class TorchDenseModel(CustomClassifier):
    def __init__(self, input_dim=None, hidden_dim=32, lr=0.01, epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _build_model(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        ).to(self.device)
        
    def fit(self, X, y):
        if self.model is None:
            num_classes = len(np.unique(y))
            self.model = self._build_model(X.shape[1], num_classes)
            
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        return self
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()

# Use in pipeline
pipeline = Pipeline()
pipeline.preprocess("scaler", StandardScaler())
pipeline.model("torch_model", TorchDenseModel(hidden_dim=64, epochs=50))
pipeline.metric("accuracy", "accuracy_score", y_true=y_test)

predictions, executed_graph = pipeline(X, y)
```

## Pipeline Visualization

Generate HTML reports with graphs and metrics.

```python
from accelera.src.core.pipeline import Pipeline
from accelera.src.wrappers.graph_report import GraphReport
import matplotlib.pyplot as plt

# Define custom plot function
def plot_confusion_matrix(value):
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=value)
    disp.plot(cmap="Blues")
    return plt

# Create pipeline with visualization
pipeline = Pipeline()
pipeline.preprocess("scaler", StandardScaler())
pipeline.model("model", RandomForestClassifier())
pipeline.predict("predict", X_test)
pipeline.metric(
    "confusion_matrix",
    "confusion_matrix",
    y_true=y_test,
    plot_func=plot_confusion_matrix,
    headers_name=["class_0", "class_1"]
)

# Execute
predictions, executed_graph = pipeline(X, y)

# Generate report
report = GraphReport(executed_graph)
report.generate("pipeline_report.html")
print("Report generated: pipeline_report.html")
```

## Serialization

Save and load pipelines.

```python
from accelera.src.utils.accelera_utils import serialize
from accelera.src.core.pipeline import Pipeline
import pickle

# Create and execute pipeline
pipeline = Pipeline()
pipeline.preprocess("scaler", StandardScaler())
pipeline.model("model", RandomForestClassifier())

predictions, executed_graph = pipeline(X, y)

# Serialize executed graph
serialized = serialize(executed_graph)

# Save to file
with open("pipeline.pkl", "wb") as f:
    pickle.dump(serialized, f)

# Load from file
with open("pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# Use loaded pipeline
new_predictions = loaded_pipeline.predict(X_test)
```

## Next Steps

- Explore [Core Pipeline API](core-pipeline.md#api-reference) for complete documentation
- Learn about [Code Parallelizer](code-parallelizer.md#api-reference) features
- See [Contributing](contributing.md) to add your own examples

A simple end-to-end ML pipeline.

```python
import sys
sys.path.insert(0, 'build/bindings')
from graph import Graph

# Create pipeline
graph = Graph()

# Load data
input_node = graph.add_input_node("data/iris.csv")

# Preprocess
preprocess_node = graph.add_preprocess_node(input_node, "standardize")

# Train model
model_node = graph.add_model_node(preprocess_node, "random_forest", 
                                   n_estimators=100)

# Evaluate
metric_node = graph.add_metric_node(model_node, "accuracy")

# Execute
graph.execute()

# Get results
accuracy = graph.get_results(metric_node)
print(f"Model Accuracy: {accuracy:.2%}")

# Generate report
graph.generate_report("report.html")
```

## Classification Task

Binary classification with the Titanic dataset.

```python
import sys
sys.path.insert(0, 'build/bindings')
from graph import Graph

# Create graph
graph = Graph()

# Load Titanic dataset
data_node = graph.add_input_node("examples/Titanic-Dataset.csv")

# Preprocessing pipeline
# 1. Handle missing values
clean_node = graph.add_preprocess_node(data_node, "handle_missing", 
                                        strategy="drop")

# 2. One-hot encode categorical variables
encoded_node = graph.add_preprocess_node(clean_node, "one_hot", 
                                          columns=["Sex", "Embarked"])

# 3. Normalize numeric features
normalized_node = graph.add_preprocess_node(encoded_node, "normalize")

# Train multiple models
rf_model = graph.add_model_node(normalized_node, "random_forest",
                                 n_estimators=200, max_depth=15)

gb_model = graph.add_model_node(normalized_node, "gradient_boosting",
                                 learning_rate=0.1, n_estimators=100)

# Evaluate both models
rf_metrics = {
    "accuracy": graph.add_metric_node(rf_model, "accuracy"),
    "precision": graph.add_metric_node(rf_model, "precision"),
    "recall": graph.add_metric_node(rf_model, "recall"),
    "f1": graph.add_metric_node(rf_model, "f1_score")
}

gb_metrics = {
    "accuracy": graph.add_metric_node(gb_model, "accuracy"),
    "precision": graph.add_metric_node(gb_model, "precision"),
    "recall": graph.add_metric_node(gb_model, "recall"),
    "f1": graph.add_metric_node(gb_model, "f1_score")
}

# Execute pipeline
graph.execute()

# Compare results
print("Random Forest Results:")
for metric, node_id in rf_metrics.items():
    value = graph.get_results(node_id)
    print(f"  {metric}: {value:.4f}")

print("\nGradient Boosting Results:")
for metric, node_id in gb_metrics.items():
    value = graph.get_results(node_id)
    print(f"  {metric}: {value:.4f}")

# Generate comparative report
graph.generate_report("titanic_comparison.html")
```

## Regression Task

Predict housing prices.

```python
import sys
sys.path.insert(0, 'build/bindings')
from graph import Graph

# Create graph
graph = Graph()

# Load housing dataset
data_node = graph.add_input_node("examples/Housing.csv")

# Feature engineering
# 1. Standardize features
scaled_node = graph.add_preprocess_node(data_node, "standardize")

# 2. PCA for dimensionality reduction (optional)
pca_node = graph.add_preprocess_node(scaled_node, "pca", 
                                      n_components=10)

# Train regression model
model_node = graph.add_model_node(pca_node, "gradient_boosting",
                                   n_estimators=150,
                                   learning_rate=0.05,
                                   max_depth=8)

# Evaluate with multiple metrics
mae_node = graph.add_metric_node(model_node, "mae")
rmse_node = graph.add_metric_node(model_node, "rmse")
r2_node = graph.add_metric_node(model_node, "r2_score")

# Execute
graph.execute()

# Display results
print("Regression Metrics:")
print(f"  MAE:  {graph.get_results(mae_node):.2f}")
print(f"  RMSE: {graph.get_results(rmse_node):.2f}")
print(f"  R²:   {graph.get_results(r2_node):.4f}")

# Save report
graph.generate_report("housing_regression.html")
```

## Loop Extraction

Extract loops from C/C++ code (Linux only).

```python
import sys
sys.path.insert(0, 'build/bindings')
from code_parallelizer_utils import extract_loops, write_loops_to_json

# Extract loops from example file
cpp_file = "examples/test_loops.cpp"
loops = extract_loops(cpp_file, ["-std=c++17"])

print(f"Found {len(loops)} loop(s) in {cpp_file}:\n")

# Display each loop
for i, loop in enumerate(loops, 1):
    print(f"Loop {i}:")
    print(f"  Type: {loop.type}")
    print(f"  Lines: {loop.start_line}-{loop.end_line}")
    
    # Show first 3 lines of code
    code_lines = loop.code.split('\\n')
    for line in code_lines[:3]:
        print(f"    {line}")
    if len(code_lines) > 3:
        print(f"    ... ({len(code_lines) - 3} more lines)")
    print()

# Save to JSON
output_file = "extracted_loops.json"
write_loops_to_json(loops, output_file)
print(f"Saved loop information to {output_file}")

# Process JSON for further analysis
import json
with open(output_file, 'r') as f:
    loop_data = json.load(f)
    
# Analyze loop distribution
loop_types = {}
for loop in loop_data:
    loop_type = loop['type']
    loop_types[loop_type] = loop_types.get(loop_type, 0) + 1

print("\nLoop Type Distribution:")
for loop_type, count in loop_types.items():
    print(f"  {loop_type}: {count}")
```

## Multi-Input Pipeline

Merge data from multiple sources.

```python
import sys
sys.path.insert(0, 'build/bindings')
from graph import Graph

# Create graph
graph = Graph()

# Load multiple datasets
train_data = graph.add_input_node("data/train.csv")
test_data = graph.add_input_node("data/test.csv")
features_data = graph.add_input_node("data/additional_features.csv")

# Merge training data with additional features
merged_train = graph.add_merge_node([train_data, features_data], 
                                     method="join", on="id")

# Preprocess merged data
preprocessed = graph.add_preprocess_node(merged_train, "normalize")

# Train model
model = graph.add_model_node(preprocessed, "random_forest")

# Merge test data with features
merged_test = graph.add_merge_node([test_data, features_data],
                                    method="join", on="id")

# Make predictions
predictions = graph.add_predict_node(model, merged_test)

# Evaluate
accuracy = graph.add_metric_node(predictions, "accuracy")

# Execute
graph.execute()

# Results
print(f"Test Accuracy: {graph.get_results(accuracy):.2%}")
```

## Custom Preprocessing

Apply custom transformation functions.

```python
import sys
sys.path.insert(0, 'build/bindings')
from graph import Graph
import numpy as np

def custom_transform(data):
    """
    Custom preprocessing function.
    Args:
        data: Input data array
    Returns:
        Transformed data array
    """
    # Apply log transformation
    data = np.log1p(data)
    
    # Remove outliers (3 standard deviations)
    mean = np.mean(data)
    std = np.std(data)
    data = data[(data > mean - 3*std) & (data < mean + 3*std)]
    
    # Normalize to [0, 1]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    return data

# Create pipeline
graph = Graph()

# Load data
data_node = graph.add_input_node("data/skewed_data.csv")

# Apply custom preprocessing
custom_node = graph.add_preprocess_node(data_node, "custom",
                                         func=custom_transform)

# Continue with standard pipeline
model_node = graph.add_model_node(custom_node, "random_forest")
metric_node = graph.add_metric_node(model_node, "accuracy")

# Execute
graph.execute()

print(f"Accuracy with custom preprocessing: {graph.get_results(metric_node):.2%}")
```

## Batch Processing

Process multiple files in parallel.

```python
import sys
sys.path.insert(0, 'build/bindings')
from code_parallelizer_utils import extract_loops, write_loops_to_json
from multiprocessing import Pool
import os

def process_file(cpp_file):
    """Process a single C++ file."""
    try:
        loops = extract_loops(cpp_file, ["-std=c++17"])
        output_file = cpp_file.replace('.cpp', '_loops.json')
        write_loops_to_json(loops, output_file)
        return (cpp_file, len(loops), "Success")
    except Exception as e:
        return (cpp_file, 0, f"Error: {e}")

# Get all C++ files in directory
cpp_files = [f for f in os.listdir("source_code") 
             if f.endswith('.cpp') or f.endswith('.c')]

# Process in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_file, cpp_files)

# Display results
print("Batch Processing Results:")
print("-" * 60)
for filepath, loop_count, status in results:
    print(f"{filepath:30} {loop_count:3} loops  {status}")

total_loops = sum(count for _, count, _ in results)
print("-" * 60)
print(f"Total: {len(cpp_files)} files, {total_loops} loops extracted")
```

## Complete ML Workflow

Full workflow from data loading to model deployment.

```python
import sys
sys.path.insert(0, 'build/bindings')
from graph import Graph

# Initialize
graph = Graph()

# 1. Data Loading
print("Step 1: Loading data...")
data_node = graph.add_input_node("data/dataset.csv")

# 2. Data Preprocessing
print("Step 2: Preprocessing...")
clean_node = graph.add_preprocess_node(data_node, "handle_missing")
scaled_node = graph.add_preprocess_node(clean_node, "standardize")

# 3. Feature Engineering
print("Step 3: Feature engineering...")
pca_node = graph.add_preprocess_node(scaled_node, "pca", n_components=50)

# 4. Model Training
print("Step 4: Training models...")
models = {
    "RF": graph.add_model_node(pca_node, "random_forest", n_estimators=200),
    "GB": graph.add_model_node(pca_node, "gradient_boosting"),
    "NN": graph.add_model_node(pca_node, "neural_network", layers=[64, 32])
}

# 5. Model Evaluation
print("Step 5: Evaluating models...")
results = {}
for name, model_id in models.items():
    results[name] = {
        "accuracy": graph.add_metric_node(model_id, "accuracy"),
        "f1": graph.add_metric_node(model_id, "f1_score")
    }

# 6. Execute Pipeline
print("Step 6: Executing pipeline...")
graph.execute()

# 7. Compare Results
print("\nStep 7: Results:")
print("=" * 50)
for model_name, metrics in results.items():
    print(f"\n{model_name} Model:")
    for metric_name, node_id in metrics.items():
        value = graph.get_results(node_id)
        print(f"  {metric_name}: {value:.4f}")

# 8. Generate Report
print("\nStep 8: Generating report...")
graph.generate_report("complete_workflow_report.html")
print("✓ Report saved to complete_workflow_report.html")

# 9. Export Pipeline
graph.export_graph("pipeline_structure.png")
print("✓ Pipeline diagram saved to pipeline_structure.png")
```

## Next Steps

- Check [Core Pipeline API](core-pipeline.md#api-reference) for complete function documentation
- See [Code Parallelizer](code-parallelizer.md#api-reference) for advanced loop extraction features
- Read [Contributing](contributing.md) to add your own examples
