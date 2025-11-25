#!/bin/bash

# MLX ROCm Setup Script
# Installs required ROCm packages for cross-compiling MLX

set -e

echo "Setting up ROCm development environment for MLX..."
echo ""

# Check if running on Arch Linux
if ! grep -q "Arch Linux" /etc/os-release; then
    echo "Warning: This script is designed for Arch Linux."
    echo "Please install ROCm packages manually for your distribution."
    exit 1
fi

echo "Installing ROCm packages..."

# Install ROCm development packages
sudo pacman -S --needed \
    hip-runtime-amd \
    rocblas \
    rocm-hip-sdk \
    clang \
    cmake

echo ""
echo "ROCm packages installed successfully!"
echo ""
echo "Next steps:"
echo "1. Run './build_rocm.sh' to build MLX with ROCm support"
echo "2. The build artifacts will be in the 'rocm-build/' directory"
echo "3. Test on an AMD GPU system by copying the built binaries"
echo ""
echo "Environment variables for runtime (on AMD GPU systems):"
echo "  export HSA_OVERRIDE_GFX_VERSION=10.3.0"
echo "  export AMDGPU_TARGETS=gfx1030"
echo "  export GPU_TARGETS=gfx1030"
