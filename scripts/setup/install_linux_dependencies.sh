#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
    echo "This helper currently supports Ubuntu and Debian systems with apt-get." >&2
    exit 1
fi

# The resident Slang compiler uses C++20 features missing from Jammy's Clang.
# Keep building against the runner's system libraries with a current compiler.
llvm_version=22
. /etc/os-release
llvm_key=/etc/apt/keyrings/infernux-llvm.asc
sudo install -d -m 0755 /etc/apt/keyrings
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee "$llvm_key" >/dev/null
printf 'deb [signed-by=%s] https://apt.llvm.org/%s/ llvm-toolchain-%s-%s main\n' \
    "$llvm_key" "$VERSION_CODENAME" "$VERSION_CODENAME" "$llvm_version" | \
    sudo tee /etc/apt/sources.list.d/infernux-llvm.list >/dev/null

sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
    build-essential \
    ccache \
    "clang-$llvm_version" \
    "clang-format-$llvm_version" \
    git \
    libasound2-dev \
    libdecor-0-dev \
    libdrm-dev \
    libegl-dev \
    libegl1 \
    libffi-dev \
    libgbm-dev \
    libgl1 \
    libpipewire-0.3-dev \
    libpulse-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    libvulkan-dev \
    libwayland-dev \
    libx11-dev \
    libx11-xcb1 \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libxcursor-dev \
    libxext-dev \
    libxfixes-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-0 \
    libxrandr-dev \
    libxrender-dev \
    libxss-dev \
    libxtst-dev \
    "lld-$llvm_version" \
    "llvm-$llvm_version" \
    mesa-vulkan-drivers \
    ninja-build \
    pkg-config \
    spirv-tools \
    vulkan-validationlayers \
    vulkan-tools \
    xauth \
    xvfb \
    zlib1g-dev

llvm_bin="/usr/lib/llvm-$llvm_version/bin"
if [[ -n ${GITHUB_PATH:-} ]]; then
    printf '%s\n' "$llvm_bin" >> "$GITHUB_PATH"
else
    printf 'Run with this LLVM toolchain: export PATH="%s:$PATH"\n' "$llvm_bin"
fi
echo "Linux native build dependencies are installed."
