# Contributing to Accelera

Thank you for your interest in contributing to Accelera! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project guidelines

## Getting Started

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/accelera.git
cd accelera
```

### 2. Add Upstream Remote

```bash
git remote add upstream https://github.com/Mohamed-Ashraf273/accelera.git
```

### 3. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

## Development Setup

### Prerequisites

- Python 3.9+
- CMake 3.14+
- C++20 compiler
- LLVM/Clang 14+ (Linux only, for code parallelizer)

### Install Dependencies

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Linux/macOS

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pre-commit pytest black flake8 mypy
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

### Build the Project

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTS=ON
cmake --build . -j4
```

## Making Changes

### Project Structure

```
accelera/
├── accelera/           # Python package
│   ├── api/           # High-level API
│   ├── bindings/      # pybind11 bindings
│   └── src/           # Python source
├── build/             # Build artifacts
├── docs/              # Documentation
├── examples/          # Example scripts
├── include/           # C++ headers
│   ├── core/         # Core functionality
│   ├── nodes/        # Node implementations
│   └── utils/        # Utilities
├── src/               # C++ source
│   ├── core/         # Core implementations
│   ├── nodes/        # Node implementations
│   └── utils/        # Utility implementations
├── shell/             # Shell scripts
└── tests/             # Test files
```

### Adding a New Feature

1. **Plan**: Discuss in an issue first
2. **Branch**: Create a feature branch
3. **Implement**: Write code following style guidelines
4. **Test**: Add tests for your feature
5. **Document**: Update relevant documentation
6. **Format**: Run pre-commit checks

### Adding a New Node Type

Example: Adding a `FilterNode`

**1. Header file** (`include/nodes/filter.hpp`):

```cpp
#ifndef ACCELERA_FILTER_NODE_HPP
#define ACCELERA_FILTER_NODE_HPP

#include "core/node.hpp"

namespace accelera {

class FilterNode : public Node {
public:
    FilterNode(const std::string& condition);
    void execute() override;
    
private:
    std::string condition_;
};

} // namespace accelera

#endif
```

**2. Implementation** (`src/nodes/filter.cpp`):

```cpp
#include "nodes/filter.hpp"

namespace accelera {

FilterNode::FilterNode(const std::string& condition)
    : Node("filter"), condition_(condition) {}

void FilterNode::execute() {
    // Implementation
}

} // namespace accelera
```

**3. Python binding** (`accelera/bindings/filter.cpp`):

```cpp
#include <pybind11/pybind11.h>
#include "nodes/filter.hpp"

namespace py = pybind11;

PYBIND11_MODULE(filter, m) {
    py::class_<accelera::FilterNode>(m, "FilterNode")
        .def(py::init<const std::string&>())
        .def("execute", &accelera::FilterNode::execute);
}
```

**4. Add to CMakeLists.txt**:

```cmake
list(APPEND NODE_SOURCES "${CMAKE_SOURCE_DIR}/src/nodes/filter.cpp")
```

**5. Add tests** (`tests/test_filter.py`):

```python
import pytest
from filter import FilterNode

def test_filter_creation():
    node = FilterNode("value > 10")
    assert node is not None

def test_filter_execution():
    node = FilterNode("value > 10")
    node.execute()
    # Add assertions
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_graph.py

# Run with coverage
pytest --cov=accelera
```

### Writing Tests

**Python tests:**

```python
import pytest
from graph import Graph

def test_graph_creation():
    """Test that a graph can be created."""
    graph = Graph()
    assert graph is not None

def test_add_input_node():
    """Test adding an input node."""
    graph = Graph()
    node_id = graph.add_input_node("test.csv")
    assert node_id >= 0
```


## Code Style

### Python

Just run:

```bash
pre-commit run --files --hook-stage manual
```
for formatting

### Documentation

- Update relevant `.md` files in `docs/`
- Add docstrings to Python code
- Add Doxygen comments to C++ code

**C++ Doxygen example:**

```cpp
/**
 * @brief Process input data
 * 
 * @param data Input data vector
 * @param normalize Whether to normalize
 * @return Processed data
 */
std::vector<double> process_data(
    const std::vector<double>& data,
    bool normalize = true);
```

## Submitting Changes

### Before Submitting

1. **Run all tests:**
   ```bash
   pytest
   ```

2. **Run pre-commit checks:**
   ```bash
   pre-commit run --all-files
   ```

3. **Build successfully:**
   ```bash
   cd build && cmake --build . -j4
   ```

4. **Update documentation** if needed

### Commit Messages

Follow conventional commits:

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
git commit -m "feat(graph): add FilterNode for data filtering"
git commit -m "fix(parallelizer): handle code fragments correctly"
git commit -m "docs: update installation guide with LLVM setup"
```

### Creating a Pull Request

1. **Push to your fork:**
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Create PR on GitHub:**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

3. **PR Description should include:**
   - What changes were made
   - Why they were made
   - How to test them
   - Related issues (if any)

### Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Additional Resources

- [GitHub Issues](https://github.com/Mohamed-Ashraf273/accelera/issues)
- [Discussions](https://github.com/Mohamed-Ashraf273/accelera/discussions)
- [Documentation](https://accelera.readthedocs.io)

## Questions?

If you have questions:
- Open a [GitHub Discussion](https://github.com/Mohamed-Ashraf273/accelera/discussions)
- Check existing [Issues](https://github.com/Mohamed-Ashraf273/accelera/issues)
- Ask in the PR comments

Thank you for contributing to Accelera! 🚀
