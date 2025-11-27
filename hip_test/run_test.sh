#!/bin/bash
# Quick test script for HIP functionality

echo "=== HIP Test Runner ==="
echo "This will test basic HIP functionality on your Steam Deck"
echo ""

# Set environment
export HSA_OVERRIDE_GFX_VERSION=10.3.0
echo "Set HSA_OVERRIDE_GFX_VERSION=10.3.0"

# Check if already built
if [ ! -f "simple_hip_test" ]; then
    echo "Building test..."
    make
    if [ $? -ne 0 ]; then
        echo "❌ Build failed!"
        exit 1
    fi
    echo "✅ Build successful"
    echo ""
fi

echo "Running HIP test..."
echo "========================================"
./simple_hip_test
exit_code=$?
echo "========================================"

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 HIP test PASSED! Your ROCm setup is working correctly."
    echo "If MLX still fails, the issue is in the MLX integration, not HIP itself."
else
    echo ""
    echo "❌ HIP test FAILED! Check your ROCm installation and GPU setup."
    echo "Run 'rocminfo' to verify GPU detection."
fi

exit $exit_code
