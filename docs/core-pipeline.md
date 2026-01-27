# Core Pipeline Module

The Core Pipeline Module is Accelera's graph-based architecture for building and executing machine learning workflows.

## Overview

The Core Pipeline Module provides a high-performance, flexible framework for constructing ML pipelines using a directed acyclic graph (DAG) structure. It combines Python's ease of use with C++'s performance for compute-intensive operations.

## Key Features

- **Graph-Based Architecture**: Build ML pipelines as directed acyclic graphs
- **Parallel Execution**: Execute independent branches in parallel
- **Flexible Design**: Chain preprocessing, models, and metrics easily
- **Branch Support**: Create parallel pipeline paths for experimentation
- **Python API**: Intuitive Python interface with C++ backend
- **Visualization**: Generate HTML reports with pipeline visualizations

## Quick Start

### Basic Pipeline

```python
from accelera.src.core.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create pipeline
pipeline = Pipeline()

# Add preprocessing
pipeline.preprocess("scaler", StandardScaler())

# Add model
pipeline.model("rf", RandomForestClassifier(n_estimators=100))

# Add metric
pipeline.metric("accuracy", "accuracy_score")

# Execute pipeline
predictions, executed_graph = pipeline(X, y)

print(f"Accuracy: {predictions[0]:.4f}")
```

## Core Concepts

### Pipeline Structure

The Pipeline class manages the execution flow:

```python
from accelera.src.core.pipeline import Pipeline

# Create pipeline
p = Pipeline()
```

### Preprocessing

Add preprocessing steps to transform data:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Single preprocessing step
p.preprocess("scaler", StandardScaler())

# Multiple preprocessing steps
p.preprocess("scaler", StandardScaler())
p.preprocess("normalizer", MinMaxScaler())
```

### Branching

Create parallel paths for experimentation:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Branch preprocessing
p.branch(
    "preprocessing",
    p.preprocess("standard_scaler", StandardScaler(), branch=True),
    p.preprocess("min_max_scaler", MinMaxScaler(), branch=True),
)

# Branch models
p.branch(
    "models",
    p.model("lr", LogisticRegression(), branch=True),
    p.model("rf", RandomForestClassifier(), branch=True),
    p.model("svm", SVC(), branch=True),
)
```

### Models

Train machine learning models:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Single model
p.model("rf", RandomForestClassifier(n_estimators=100))

# Multiple models in branches
p.branch(
    "models",
    p.model("rf", RandomForestClassifier(n_estimators=100), branch=True),
    p.model("gb", GradientBoostingClassifier(learning_rate=0.1), branch=True),
)
```

### Metrics

Evaluate model performance:

```python
# Single metric
p.metric("accuracy", "accuracy_score")

# Multiple metrics in branches
p.branch(
    "metrics",
    p.metric("accuracy", "accuracy_score", branch=True),
    p.metric("precision", "precision_score", branch=True),
    p.metric("f1", "f1_score", branch=True),
)
```

### Predictions

Make predictions on test data:

```python
# Add prediction step
p.predict("predict", test_data=X_test)

# Then add metrics
p.metric("accuracy", "accuracy_score", y_true=y_test)
```

## Advanced Usage

### Complete Branching Example

Compare multiple preprocessing methods and models:

```python
import numpy as np
from accelera.src.core.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_test, y_test = make_classification(n_samples=200, n_features=20, random_state=43)

# Create pipeline
p = Pipeline()

# Branch preprocessing (2 branches)
p.branch(
    "preprocessing",
    p.preprocess("standard_scaler", StandardScaler(), branch=True),
    p.preprocess("power_transform", 
                 lambda x: np.sign(x) * np.power(np.abs(x), 0.8), 
                 branch=True),
)

# Common preprocessing
p.preprocess("clip", lambda x: np.clip(x, -5, 5))

# Branch models (2 branches)
p.branch(
    "models",
    p.model("logreg", LogisticRegression(max_iter=1000), branch=True),
    p.model("rf", RandomForestClassifier(n_estimators=100), branch=True),
)

# Predict
p.predict("predict", X_test)

# Branch metrics (2 branches)
p.branch(
    "metrics",
    p.metric("accuracy", "accuracy_score", y_true=y_test, branch=True),
    p.metric("precision", "precision_score", y_true=y_test, branch=True),
)

# Execute - runs 8 pipelines in parallel:
# 2 preprocessors × 2 models × 2 metrics = 8 combinations
predictions, executed_graph = p(X, y)

print(f"Total pipelines executed: {len(predictions)}")
```

### Custom Models

Use custom models by implementing the CustomClassifier interface:

```python
import torch
import torch.nn as nn
from accelera.src.custom.classifier import CustomClassifier

class TorchModel(CustomClassifier):
    def __init__(self, hidden_dim=32, lr=0.01, epochs=100):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _build_model(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        ).to(self.device)
    
    def fit(self, X, y):
        # Training implementation
        self.model = self._build_model(X.shape[1], len(np.unique(y)))
        # ... training code ...
        return self
    
    def predict(self, X):
        # Prediction implementation
        # ... prediction code ...
        return predictions

# Use in pipeline
p.model("torch", TorchModel(hidden_dim=64))
```

### Report Generation

Generate HTML reports with visualizations:

```python
from accelera.src.wrappers.graph_report import GraphReport

# After pipeline execution
predictions, executed_graph = pipeline(X, y)

# Generate report
report = GraphReport("pipeline_report", executed_graph)
report.execute()

print("Report saved to: pipeline_report.html")
```

## Performance

### Parallel Execution

The branching feature automatically parallelizes independent paths:

```python
import time

# Sequential sklearn pipeline would run N pipelines sequentially
# Accelera runs them in parallel

start = time.time()
predictions, executed_graph = pipeline(X, y)
parallel_time = time.time() - start

print(f"Parallel execution time: {parallel_time:.2f}s")
```

### Memory Efficiency

Accelera uses C++ backend for compute-intensive operations, reducing memory overhead compared to pure Python implementations.

## Examples

### Example 1: Model Comparison

**File**: `examples/demo.py`

Compare multiple models on the same dataset:

```python
from accelera.src.core.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

pipeline = Pipeline()

pipeline.branch(
    "preprocessing",
    pipeline.preprocess("scaler", StandardScaler(), branch=True),
).branch(
    "models",
    pipeline.model("rf", RandomForestClassifier(n_estimators=100), branch=True),
    pipeline.model("lr", LogisticRegression(max_iter=1000), branch=True),
    pipeline.model("svm", SVC(kernel='rbf'), branch=True),
).metric("accuracy", "accuracy_score")

predictions, executed_graph = pipeline(X, y)
```

### Example 2: Sklearn Comparison

**File**: `examples/sklearn_comp.py`

Compare Accelera's parallel execution with sklearn's sequential pipeline.

### Example 3: Classification with Reports

**File**: `examples/report_model.py`

Complete classification workflow with the Titanic dataset including model reporting.

## API Reference

### Pipeline Class

#### `__init__()`

Create a new pipeline.

#### `preprocess(name: str, transformer, branch: bool = False)`

Add preprocessing step.

**Parameters:**
- `name`: Unique identifier for this step
- `transformer`: sklearn transformer or callable
- `branch`: Whether this is a branch step

#### `model(name: str, estimator, branch: bool = False)`

Add model training step.

**Parameters:**
- `name`: Unique identifier for this model
- `estimator`: sklearn estimator or custom model
- `branch`: Whether this is a branch step

#### `predict(name: str, test_data)`

Add prediction step.

**Parameters:**
- `name`: Unique identifier for prediction step
- `test_data`: Test data array

#### `metric(name: str, metric_name: str, y_true=None, branch: bool = False)`

Add evaluation metric.

**Parameters:**
- `name`: Unique identifier for metric
- `metric_name`: Metric function name (e.g., "accuracy_score")
- `y_true`: True labels for evaluation
- `branch`: Whether this is a branch step

#### `branch(name: str, *steps)`

Create parallel branches.

**Parameters:**
- `name`: Name for this branching point
- `*steps`: Variable number of pipeline steps to execute in parallel

#### `__call__(X, y)`

Execute the pipeline.

**Parameters:**
- `X`: Training features
- `y`: Training labels

**Returns:**
- `predictions`: List of prediction results
- `executed_graph`: Graph object with execution details

## Troubleshooting

### Common Issues

**Pipeline execution fails:**
- Ensure all steps are properly connected
- Check that data types are compatible
- Verify sklearn estimators are properly initialized

**Poor performance:**
- Try different preprocessing combinations
- Adjust model hyperparameters
- Use branching to compare multiple approaches

**Memory issues:**
- Reduce dataset size for testing
- Limit number of branches
- Use simpler models initially

## API Reference

Complete reference for the Core Pipeline Module based on actual implementation.

### Pipeline Class

**Import:**
```python
from accelera.src.core.pipeline import Pipeline
```

The main class for building ML pipelines with branching support.

#### Constructor

```python
Pipeline()
```

Creates a new empty pipeline.

**Example:**
```python
pipeline = Pipeline()
```

---

#### `preprocess(name: str, transformer, branch: bool = False)`

Add a preprocessing step to the pipeline.

**Parameters:**
- `name` (str): Unique identifier for this preprocessing step
- `transformer`: sklearn transformer or callable function
- `branch` (bool, optional): If True, creates a branch. Default is False

**Returns:**
- `self`: Returns the pipeline instance for method chaining

**Supported Transformers:**
- sklearn transformers: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, etc.
- Custom callables: Any function that transforms the data

**Example:**
```python
from sklearn.preprocessing import StandardScaler

# Single preprocessing
pipeline.preprocess("scaler", StandardScaler())

# Custom preprocessing function
pipeline.preprocess("clip", lambda x: np.clip(x, -5, 5))

# Branching preprocessing
pipeline.preprocess("standard", StandardScaler(), branch=True)
pipeline.preprocess("minmax", MinMaxScaler(), branch=True)
```

---

#### `model(name: str, estimator, branch: bool = False)`

Add a model training step to the pipeline.

**Parameters:**
- `name` (str): Unique identifier for this model
- `estimator`: sklearn estimator or custom model implementing fit/predict
- `branch` (bool, optional): If True, creates a branch. Default is False

**Returns:**
- `self`: Returns the pipeline instance for method chaining

**Supported Models:**
- sklearn estimators: `RandomForestClassifier`, `LogisticRegression`, `SVC`, etc.
- Custom models: Inherit from `CustomClassifier` or implement fit/predict methods

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Single model
pipeline.model("rf", RandomForestClassifier(n_estimators=100))

# Branching models
pipeline.model("rf", RandomForestClassifier(n_estimators=100), branch=True)
pipeline.model("lr", LogisticRegression(max_iter=1000), branch=True)
```

---

#### `predict(name: str, test_data: np.ndarray)`

Add a prediction step for evaluation on test data.

**Parameters:**
- `name` (str): Unique identifier for this prediction step
- `test_data` (np.ndarray): Test features for prediction

**Returns:**
- `self`: Returns the pipeline instance for method chaining

**Example:**
```python
# Add prediction step
pipeline.predict("predict", test_data=X_test)
```

---

#### `metric(name: str, metric_name: str, y_true=None, branch: bool = False)`

Add an evaluation metric to the pipeline.

**Parameters:**
- `name` (str): Unique identifier for this metric
- `metric_name` (str): Name of the sklearn metric (e.g., "accuracy_score", "f1_score")
- `y_true` (array-like, optional): True labels for evaluation (required for test metrics)
- `branch` (bool, optional): If True, creates a branch. Default is False

**Returns:**
- `self`: Returns the pipeline instance for method chaining

**Supported Metrics:**
- Classification: `"accuracy_score"`, `"precision_score"`, `"recall_score"`, `"f1_score"`
- Regression: `"mean_squared_error"`, `"r2_score"`, `"mean_absolute_error"`

**Example:**
```python
# Single metric
pipeline.metric("accuracy", "accuracy_score")

# With test data
pipeline.metric("accuracy", "accuracy_score", y_true=y_test)

# Branching metrics
pipeline.metric("accuracy", "accuracy_score", y_true=y_test, branch=True)
pipeline.metric("f1", "f1_score", y_true=y_test, branch=True)
```

---

#### `branch(name: str, *steps)`

Create parallel branches in the pipeline.

**Parameters:**
- `name` (str): Name for this branching point
- `*steps`: Variable number of pipeline steps (with `branch=True`) to execute in parallel

**Returns:**
- `self`: Returns the pipeline instance for method chaining

**Example:**
```python
# Branch preprocessing
pipeline.branch(
    "preprocessing",
    pipeline.preprocess("standard", StandardScaler(), branch=True),
    pipeline.preprocess("minmax", MinMaxScaler(), branch=True),
)

# Branch models
pipeline.branch(
    "models",
    pipeline.model("rf", RandomForestClassifier(), branch=True),
    pipeline.model("lr", LogisticRegression(), branch=True),
)
```

---

#### `__call__(X: np.ndarray, y: np.ndarray)`

Execute the complete pipeline.

**Parameters:**
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels

**Returns:**
- `predictions` (list): List of prediction results from all pipeline paths
- `executed_graph`: Graph object containing execution details and results

**Example:**
```python
# Execute pipeline
predictions, executed_graph = pipeline(X_train, y_train)

# Access results
print(f"Number of pipelines executed: {len(predictions)}")
print(f"First pipeline result: {predictions[0]}")
```

---

### CustomClassifier Class

**Import:**
```python
from accelera.src.custom.classifier import CustomClassifier
```

Base class for implementing custom models (e.g., PyTorch, TensorFlow).

#### Required Methods

##### `fit(X, y)`

Train the model.

**Parameters:**
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels

**Returns:**
- `self`: The fitted model instance

##### `predict(X)`

Make predictions.

**Parameters:**
- `X` (np.ndarray): Features for prediction

**Returns:**
- `predictions` (np.ndarray): Predicted labels

##### `predict_proba(X)` (optional)

Get probability estimates.

**Parameters:**
- `X` (np.ndarray): Features for prediction

**Returns:**
- `probabilities` (np.ndarray): Probability estimates for each class

**Example:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from accelera.src.custom.classifier import CustomClassifier

class TorchModel(CustomClassifier):
    def __init__(self, hidden_dim=32, lr=0.01, epochs=100):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def _build_model(self, input_dim, num_classes):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes),
        ).to(self.device)
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        
        n_features = X.shape[1]
        num_classes = len(np.unique(y))
        
        self.model = self._build_model(n_features, num_classes)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        X_tensor = torch.tensor(X, device=self.device)
        y_tensor = torch.tensor(y, device=self.device)
        
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        return preds
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        
        return probabilities

# Use in pipeline
pipeline.model("torch", TorchModel(hidden_dim=64, epochs=50))
```

---

### GraphReport Class

**Import:**
```python
from accelera.src.wrappers.graph_report import GraphReport
```

Generate HTML reports with pipeline visualizations and metrics.

#### Constructor

```python
GraphReport(report_name: str, executed_graph)
```

**Parameters:**
- `report_name` (str): Name for the output HTML file (without extension)
- `executed_graph`: The executed graph object returned from pipeline execution

#### `execute()`

Generate and save the HTML report.

**Example:**
```python
from accelera.src.wrappers.graph_report import GraphReport

# After pipeline execution
predictions, executed_graph = pipeline(X, y)

# Generate report
report = GraphReport("pipeline_results", executed_graph)
report.execute()

print("Report saved to: pipeline_results.html")
```

The report includes:
- Pipeline structure visualization
- Execution metrics for each step
- Performance comparisons across branches
- Timing information

---

### Complete Example from examples/demo.py

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from accelera.src.core.pipeline import Pipeline
from accelera.src.wrappers.graph_report import GraphReport

# Generate data
X, y = make_classification(
    n_samples=10000,
    n_features=25,
    n_classes=4,
    random_state=42
)

test_data = X[:500]
y_test = y[:500]

# Build pipeline with branches
pipeline = Pipeline()

# Branch preprocessing (2 options)
pipeline.branch(
    "preprocessing",
    pipeline.preprocess("standard", StandardScaler(), branch=True),
    pipeline.preprocess("power", 
                       lambda x: np.sign(x) * np.power(np.abs(x), 0.8), 
                       branch=True),
)

# Common preprocessing
pipeline.preprocess("clip", lambda x: np.clip(x, -5, 5))

# Branch models (4 options)
pipeline.branch(
    "models",
    pipeline.model("rf", RandomForestClassifier(n_estimators=100), branch=True),
    pipeline.model("lr", LogisticRegression(max_iter=1000), branch=True),
    pipeline.model("svm", SVC(kernel='rbf'), branch=True),
    pipeline.model("torch", TorchDenseModel(hidden_dim=64), branch=True),
)

# Predict on test data
pipeline.predict("predict", test_data)

# Branch metrics (2 options)
pipeline.branch(
    "metrics",
    pipeline.metric("accuracy", "accuracy_score", y_true=y_test, branch=True),
    pipeline.metric("f1", "f1_score", y_true=y_test, average="macro", branch=True),
)

# Execute - runs 2×4×2 = 16 pipelines in parallel
predictions, executed_graph = pipeline(X, y)

print(f"Executed {len(predictions)} pipeline combinations")

# Generate report
report = GraphReport("demo_report", executed_graph)
report.execute()
```

---

## Best Practices

1. **Start Simple**: Begin with basic pipelines and add complexity
2. **Use Branching**: Compare multiple approaches in parallel
3. **Test Incrementally**: Validate each step before adding more
4. **Monitor Performance**: Use reports to track execution time
5. **Custom Models**: Implement CustomClassifier for non-sklearn models

## Next Steps

- Check [Examples](examples.md) for more use cases
- See [API Reference](#api-reference) section above for complete documentation
- Visit [Contributing](contributing.md) to help improve the module

## Related Modules

- [Code Parallelizer](code-parallelizer.md) - Extract loops from C/C++ code
- [AutoML Module](automl.md) - Automatic pipeline generation (Coming Soon)
- [Deployment Module](deployment.md) - Model deployment (Coming Soon)
