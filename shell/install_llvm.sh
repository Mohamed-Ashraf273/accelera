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

echo "Updating package lists..."
apt-get update

echo "Installing LLVM 14 and Clang 14 development packages..."
apt-get install -y llvm-14-dev libclang-14-dev clang-14

echo "Successfully installed LLVM/Clang 14"
