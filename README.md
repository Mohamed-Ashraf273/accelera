# Accelera 🚀

**A High-Performance Machine Learning Pipeline Framework**

Accelera is a cutting-edge ML pipeline framework that combines the flexibility of Python with the performance of C++. It provides a robust, scalable solution for building and deploying machine learning workflows with optimized performance.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Mohamed-Ashraf273/Accelera)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++17](https://img.shields.io/badge/C++-17-orange.svg)](https://isocpp.org/)

## Features

- **High Performance**: C++ backend for compute-intensive operations
- **Python Friendly**: Intuitive Python API for ease of use
- **Flexible Pipeline**: Build complex ML workflows with modular components
- **Built-in Visualization**: Generate reports and graphs automatically
- **Parallel Processing**: Optimized for multi-core execution
- **Extensible**: Easy to add custom nodes and operations

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mohamed-Ashraf273/accelera.git
cd accelera

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 🚨 **Important Note for Developers**

**Before starting any run or execution in your session, you must set the PYTHONPATH:**

**For Linux/macOS:**
```bash
export PYTHONPATH="/path/to/your/project/accelera"
```

**For Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "C:\path\to\your\project\accelera"
```

**For Windows (Command Prompt):**
```cmd
set PYTHONPATH=C:\path\to\your\project\accelera
```

### Prerequisites for Linux Users

**For Linux users, you need to install LLVM/Clang for the code parallelizer features:**

```bash
# Install LLVM 18 (recommended - latest version)
sudo bash shell/install_llvm.sh 18

# Or install LLVM 14 (stable version)
sudo bash shell/install_llvm.sh 14

# Or use default (LLVM 18)
sudo bash shell/install_llvm.sh
```

**What this installs:**
- LLVM development libraries (for AST parsing)
- Clang compiler and development headers
- Required for loop extraction and code analysis features

**Note for macOS/Windows users:** The code parallelizer is currently Linux-only. Other features will work normally.

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
pytest accelera/

# Run specific test
python -c "from accelera.src.core.pipeline_test import test_parallel_execution_performance; test_parallel_execution_performance()"

# Run heavy parallel test
python test_heavy_parallel.py
```

## Usage Example

```python
import sys
import os

# Add project to Python path
project_root = "/path/to/your/accelera/project"
sys.path.insert(0, project_root)

# Import accelera modules
from accelera.src.core.pipeline import Pipeline
from accelera.src.custom.classifier import CustomClassifier

# Create and run pipeline
pipeline = Pipeline()
# Add your ML components here...
```

### Code Optimizer Usage

```python
from accelera.src.utils.code_optimizer import CodeOptimizer

# Initialize optimizer (uses Groq API by default)
# Will prompt for API key if not set in environment
optimizer = CodeOptimizer()

# Convert Python code to C++
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
"""

cpp_code = optimizer.convert_to_cpp("fibonacci.cpp", python_code)
# Creates optimized, formatted C++ code in fibonacci.cpp
# Automatically applies clang-format for clean code style
```

## Project Structure

```
accelera/
├── build/                  # Build artifacts and C++ bindings
├── examples/              # Example scripts and demos
├── include/               # C++ header files
├── accelera/               # Main Python package
│   ├── api/              # Public API
│   ├── bindings/         # Python-C++ bindings
│   └── src/              # Source code
├── src/                   # C++ source files
├── tools/                 # Development tools
└── CMakeLists.txt        # Build configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run pre-commit checks: `pre-commit run --all-files --hook-stage manual`
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with love using C++ and Python
- Powered by CMake and pybind11
- Inspired by modern ML pipeline frameworks

---

**Happy coding! 🎉**