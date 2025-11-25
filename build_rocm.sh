#!/bin/bash

# MLX ROCm Build Script
# This script sets up the environment and builds MLX with ROCm support

set -e

echo "Setting up ROCm environment..."

# Set ROCm environment variables for cross-compilation
export HIP_PLATFORM=amd
export HIP_COMPILER=clang
export HIP_RUNTIME=rocclr
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export AMDGPU_TARGETS=gfx1030
export GPU_TARGETS=gfx1030
export ROCM_TARGETS=gfx1030
export CMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++

# Add ROCm paths if needed
export PATH="/opt/rocm/bin:/opt/rocm/hip/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/hip/lib:$LD_LIBRARY_PATH"
export CPATH="/opt/rocm/include:/opt/rocm/hip/include:$CPATH"

echo "ROCm environment variables set:"
echo "  HIP_PLATFORM=$HIP_PLATFORM"
echo "  HIP_COMPILER=$HIP_COMPILER"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  AMDGPU_TARGETS=$AMDGPU_TARGETS"

# Create build directory
BUILD_DIR="rocm-build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo "Configuring CMake with ROCm support..."

# Configure with ROCm enabled
cmake .. \
    -DMLX_BUILD_ROCM=ON \
    -DMLX_BUILD_METAL=OFF \
    -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_PYTHON_BINDINGS=OFF \
    -DMLX_BUILD_TESTS=OFF \
    -DMLX_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DHIP_PLATFORM=amd \
    -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
    -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "Building MLX with ROCm support (single-threaded for memory efficiency)..."

# Build with single thread to avoid memory issues
make -j1

echo "Build completed successfully!"
echo ""
echo "To test the build:"
echo "  cd $BUILD_DIR"
echo "  export PYTHONPATH=\"\$(pwd):\$(pwd)/../python\""
echo "  export LD_LIBRARY_PATH=\"\$(pwd)/lib:\$LD_LIBRARY_PATH\""
echo "  python -c \"import mlx.core as mx; print('ROCm backend available:', mx.rocm.is_available())\""
