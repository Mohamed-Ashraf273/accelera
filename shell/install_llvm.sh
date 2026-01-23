#!/bin/bash
set -Eeuo pipefail

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: This script only works on Linux systems"
    exit 1
fi

if [ "$EUID" -ne 0 ]; then 
    echo "This script requires sudo privileges to install packages"
    echo "Please run with: sudo $0"
    exit 1
fi

LLVM_VERSION="${1:-18}"

echo "Updating package lists..."
apt-get update

echo "Installing LLVM ${LLVM_VERSION} and Clang ${LLVM_VERSION} development packages..."
apt-get install -y llvm-${LLVM_VERSION}-dev libclang-${LLVM_VERSION}-dev clang-${LLVM_VERSION}

echo "Successfully installed LLVM/Clang ${LLVM_VERSION}"
