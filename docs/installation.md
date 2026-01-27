# Installation Guide

This guide will walk you through installing Accelera on your system.

## Prerequisites

Before installing Accelera, ensure you have the following:

- **Python 3.9 or higher**
- **CMake 3.14 or higher**
- **C++ compiler with C++20 support** (GCC 10+, Clang 10+, or MSVC 2019+)
- **Git** for cloning the repository

### Linux Additional Requirements

For code parallelizer features on Linux:
- **LLVM 14+ development libraries**
- **Clang 14+ development libraries**

## Step 1: Clone the Repository

```bash
git clone https://github.com/Mohamed-Ashraf273/accelera.git
cd accelera
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate (Linux/macOS)
source env/bin/activate

# Activate (Windows PowerShell)
.\env\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
.\env\Scripts\activate.bat
```

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Install LLVM/Clang (Linux Only)

If you're on Linux and want to use the code parallelizer features:

```bash
# Install LLVM/Clang 14
sudo ./shell/install_llvm.sh

# Or install manually
sudo apt-get update
sudo apt-get install -y llvm-14-dev libclang-14-dev clang-14
```

**Note**: Windows and macOS users can skip this step. The core ML pipeline features will work without LLVM.

## Step 5: Build the Project

```bash
# Create build directory
mkdir build
cd build

# Configure the project
cmake ..

# Build with parallel compilation (adjust -j4 based on your CPU cores)
cmake --build . -j4
```

### Build Options

You can customize the build with CMake options:

```bash
# Enable all features
cmake .. -DENABLE_PYTHON=ON -DENABLE_WHEEL=ON -DENABLE_TESTS=ON

# Build in Release mode for better performance
cmake .. -DCMAKE_BUILD_TYPE=Release

# Disable code parallelizer (force disable on Linux)
cmake .. -DLLVM_AVAILABLE=OFF
```

## Step 6: Set PYTHONPATH

**Important**: Before using Accelera, you must set the PYTHONPATH environment variable.

### Linux/macOS

```bash
export PYTHONPATH="/path/to/accelera"

# Add to ~/.bashrc or ~/.zshrc for persistence
echo 'export PYTHONPATH="/path/to/accelera"' >> ~/.bashrc
```

### Windows PowerShell

```powershell
$env:PYTHONPATH = "C:\path\to\accelera"

# For persistence (run PowerShell as Administrator)
[System.Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\path\to\accelera', [System.EnvironmentVariableTarget]::User)
```

### Windows Command Prompt

```cmd
set PYTHONPATH=C:\path\to\accelera

# For persistence
setx PYTHONPATH "C:\path\to\accelera"
```

## Step 7: Verify Installation

Test that everything is working:

```bash
# Test Python imports
python -c "import sys; sys.path.insert(0, 'build/bindings'); from graph import Graph; print('✓ Core module works')"

# Test code parallelizer (Linux only)
python examples/extract_loops.py examples/test_loops.cpp output.json
```

If you see success messages, congratulations! Accelera is installed.

## Troubleshooting

### CMake Can't Find LLVM

**Error**: `Could not find LLVM`

**Solution**: Make sure LLVM is installed:
```bash
# Check LLVM installation
llvm-config --version

# If not found, install it
sudo ./shell/install_llvm.sh
```

### Python Module Import Error

**Error**: `ModuleNotFoundError: No module named 'graph'`

**Solution**: 
1. Make sure you built the project successfully
2. Set PYTHONPATH correctly
3. Check that `build/bindings/*.so` files exist

### Build Fails with C++ Errors

**Error**: Various C++ compilation errors

**Solution**:
1. Ensure you have a C++20-compatible compiler
2. Update CMake to 3.14+
3. Check that all dependencies are installed

### Permission Denied (Linux)

**Error**: `Permission denied` when running install_llvm.sh

**Solution**:
```bash
# Make script executable
chmod +x shell/install_llvm.sh

# Run with sudo
sudo ./shell/install_llvm.sh
```

## Next Steps

Now that Accelera is installed, check out:

- [Examples](examples.md) - See sample projects
- [Core Pipeline](core-pipeline.md) - Build ML pipelines
- [Code Parallelizer](code-parallelizer.md) - Extract loops
