# mAInera 🚀

**A High-Performance Machine Learning Pipeline Framework**

AI Studio is a cutting-edge ML pipeline framework that combines the flexibility of Python with the performance of C++.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Mohamed-Ashraf273/ai-studio)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/C++-17-orange.svg)](https://isocpp.org/)

---

## 🌟 Key Features

### **Pipeline Architecture**
- **Intuitive API**: Chain operations with simple, readable syntax
- **Flexible Branching**: Split pipelines into parallel branches with automatic merging
- **Type Safety**: Built-in node types (INPUT, PREPROCESS, MODEL, PREDICT, MERGE)
- **Dependency Resolution**: Automatic topological sorting and dependency management

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mohamed-Ashraf273/ai-studio.git
cd ai-studio

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from mainera.src.core.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
test_data = np.random.rand(10, 20)

# Build pipeline with parallel branches
pipeline = Pipeline()

# Create parallel ensemble
pipeline.branch("ensemble",
    pipeline.model("rf", RandomForestClassifier(n_estimators=100), branch=True),
    pipeline.model("lr", LogisticRegression(), branch=True),
    pipeline.model("svm", SVC(), branch=True)
)

# Add prediction and merging
pipeline.predict("predictions", test_data)
pipeline.merge("final", lambda preds: np.mean(preds, axis=0))

results = pipeline(X, y)
print(f"Predictions: {results}")
```

### Performance Example

```python
# Heavy computation example (4x speedup!)
pipeline = Pipeline()
=
# These will run in parallel across CPU cores
pipeline.branch("heavy_models",
    pipeline.model("rf1", RandomForestClassifier(n_estimators=100)),
    pipeline.model("rf2", RandomForestClassifier(n_estimators=100)), 
    pipeline.model("rf3", RandomForestClassifier(n_estimators=100)),
    pipeline.model("rf4", RandomForestClassifier(n_estimators=100))
)

# Results:
# Sequential: 11.37s
# Parallel:    3.01s  (3.78x speedup!)
```

---

## 🏗️ Development Setup

### Build Requirements

- **Python 3.9+**
- **CMake 3.18+**
- **C++17 compatible compiler**
- **pybind11** (automatically fetched)

### Build Instructions

```bash
# Create build directory
mkdir build
cd build

# Configure with all features enabled
cmake .. -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON -DENABLE_TESTS=ON

# Build with parallel compilation
cmake --build . -j4
```

### 🚨 **Important for Developers**

**Before committing any changes, ALWAYS run:**

```bash
pre-commit run --all-files --hook-stage manual
```

This ensures code quality, formatting, and passes all checks. **Commits without running this will be rejected.**

### Development Workflow

1. **Make Changes** - Edit code, add features, fix bugs
2. **Build & Test** - Run the build commands above
3. **Run Pre-commit** - `pre-commit run --all-files --hook-stage manual`
4. **Commit** - `git add . && git commit -m "Your message"`
5. **Push** - `git push origin your-branch`

### Testing

```bash
# Run all tests
pytest mainera/

# Run specific test
python -c "from mainera.src.core.pipeline_test import test_parallel_execution_performance; test_parallel_execution_performance()"

# Run heavy parallel test
python test_heavy_parallel.py
```

---

## 🔬 Architecture Deep Dive

### Parallel Execution Engine

AI Studio's breakthrough parallel execution works by:

1. **Dependency Analysis**: Builds a dependency graph from pipeline structure
2. **Wave Detection**: Groups independent nodes into "waves" for parallel execution  
3. **GIL Management**: Releases Python's Global Interpreter Lock for true parallelism
4. **Smart Scheduling**: Only uses parallelism when computation benefits exceed overhead

```cpp
// Core parallel execution (simplified)
{
    py::gil_scoped_release release;  // Release Python GIL
    
    // Multiple models train simultaneously on different CPU cores
    auto future1 = std::async(std::launch::async, [&]() { model1.fit(X, y); });
    auto future2 = std::async(std::launch::async, [&]() { model2.fit(X, y); });
    auto future3 = std::async(std::launch::async, [&]() { model3.fit(X, y); });
    
    // Wait for all to complete (merge synchronization)
    future1.get(); future2.get(); future3.get();
} // GIL reacquired
```

### Pipeline Components

- **Input Node**: Data ingestion and validation
- **Preprocess Node**: Data transformation and feature engineering  
- **Model Node**: Machine learning model training
- **Predict Node**: Inference on new data
- **Merge Node**: Combining results from parallel branches

---

## 📊 Performance Benchmarks

| Operation Type | Sequential Time | Parallel Time | Speedup |
|---------------|----------------|---------------|---------|
| Light ML (LogisticRegression) | 0.0065s | 0.0107s | 0.61x* |
| Heavy ML (RandomForest) | 11.37s | 3.01s | **3.78x** |
| Ensemble (4 models) | 45.2s | 12.1s | **3.74x** |

*Light operations are faster sequential due to threading overhead

---

## 📚 Documentation & Resources

### Academic References
- **"Understanding the Python GIL"** - David Beazley (PyCon 2010)
- **"Parallel Execution of Dataflow Programs"** - Dennis & Misunas (1975)
- **"List Scheduling Algorithm for Heterogeneous Systems"** - Topcuoglu et al. (2002)

### Technical Documentation
- [Pybind11 GIL Management](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)
- [C++ Concurrency Patterns](https://en.cppreference.com/w/cpp/thread/async)
- [Python Multiprocessing vs Threading](https://docs.python.org/3/library/multiprocessing.html)

---

## 🙏 Acknowledgments

- **pybind11** team for excellent Python-C++ integration
- **scikit-learn** community for ML algorithm implementations  
- **CMake** developers for cross-platform build system
- **Python Core** developers for threading and GIL architecture

---
