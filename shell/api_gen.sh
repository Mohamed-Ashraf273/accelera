#!/bin/bash
set -Eeuo pipefail

# Get the repo root directory (two levels up from this script)
base_dir=$(dirname $(dirname "$0"))

if ! command -v pre-commit >/dev/null 2>&1
then
    echo 'Please `pip install pre-commit` to run api_gen.sh.'
    exit 1
fi

# Check if clang-format is available
if ! command -v clang-format >/dev/null 2>&1
then
    echo 'Warning: clang-format not found. C++ files will not be formatted.'
fi

# Run generate_init.py script first (relative to repo root)
echo "Generating __init__.py files..."
python3 "${base_dir}/tools/generate_init.py"

# Format code because generate_init.py might reorder imports
echo "Formatting api directory..."
(SKIP=api-gen pre-commit run --files $(find "${base_dir}"/mainera/api -type f) --hook-stage pre-commit || true) > /dev/null

# Format C++ files if clang-format is available
if command -v clang-format >/dev/null 2>&1
then
    echo "Formatting C++ files..."
    # Find all .cpp and .hpp files, excluding build directories, env, and other generated directories
    find "${base_dir}" \
        -name "*.cpp" -o -name "*.hpp" | \
        grep -vE "(/build/|/env/|/\.venv/|/cmake-build-|/CMakeFiles/|/\.git/|/__pycache__/|/\.pytest_cache/)" | \
        while read -r file; do
        echo "  Formatting $file"
        clang-format -i "$file"
    done
else
    echo "Skipping C++ formatting (clang-format not found)"
fi