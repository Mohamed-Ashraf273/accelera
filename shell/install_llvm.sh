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
LLVM_PACKAGES=(
    "llvm-${LLVM_VERSION}-dev"
    "libclang-${LLVM_VERSION}-dev"
    "clang-${LLVM_VERSION}"
)

install_llvm_packages() {
    apt-get install -y "${LLVM_PACKAGES[@]}"
}

detect_codename() {
    local codename=""

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        codename="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
    fi

    if [[ -z "$codename" ]] && command -v lsb_release >/dev/null 2>&1; then
        codename="$(lsb_release -sc)"
    fi

    echo "$codename"
}

add_llvm_apt_repository() {
    local codename="$1"
    local keyring="/usr/share/keyrings/llvm-snapshot.gpg"
    local source_file="/etc/apt/sources.list.d/llvm-toolchain-${codename}-${LLVM_VERSION}.list"

    if [[ -z "$codename" ]]; then
        echo "Could not detect Debian/Ubuntu codename for LLVM apt repository"
        return 1
    fi

    echo "Adding official LLVM apt repository for ${codename}, LLVM ${LLVM_VERSION}..."
    apt-get install -y ca-certificates wget gnupg
    rm -f "$keyring"
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o "$keyring"
    echo "deb [signed-by=${keyring}] http://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-${LLVM_VERSION} main" > "$source_file"
}

echo "Updating package lists..."
apt-get update

echo "Installing LLVM ${LLVM_VERSION} and Clang ${LLVM_VERSION} development packages..."
if ! install_llvm_packages; then
    echo "LLVM ${LLVM_VERSION} packages were not available in the current apt repositories."
    add_llvm_apt_repository "$(detect_codename)"
    apt-get update
    install_llvm_packages
fi

echo "Successfully installed LLVM/Clang ${LLVM_VERSION}"
