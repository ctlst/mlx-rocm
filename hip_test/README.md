# Simple HIP Test

This is a standalone test to verify that your ROCm/HIP setup is working correctly, independent of MLX.

## What it tests:
- ✅ HIP device detection
- ✅ GPU memory allocation (`hipMalloc`)
- ✅ Host-to-device data transfer (`hipMemcpy`)
- ✅ Kernel compilation and launching (`hipLaunchKernelGGL`)
- ✅ Device-to-host data transfer (`hipMemcpy`)
- ✅ Basic kernel execution (adds 1 to each array element)

## Quick Test (using Makefile):

```bash
cd hip_test
export HSA_OVERRIDE_GFX_VERSION=10.3.0
make
./simple_hip_test
```

Expected output:
```
=== Simple HIP Test ===
Found 1 HIP device(s)
Device: Radeon RX 6600/6600 XT/6600M
Compute capability: 10.3
Input data: 0 1 2 3 4 5 6 7 8 9
✅ GPU memory allocated
✅ Data copied to GPU
✅ Kernel launched successfully
✅ Kernel execution completed
✅ Data copied back to CPU
Output data: 1 2 3 4 5 6 7 8 9 10
✅ TEST PASSED! HIP is working correctly.
GPU memory allocation, data transfer, and kernel execution all work.
```

## Alternative (using CMake):

```bash
cd hip_test
mkdir build && cd build
cmake ..
make
export HSA_OVERRIDE_GFX_VERSION=10.3.0
./simple_hip_test
```

## Troubleshooting:

### If "No HIP devices found":
- Check `rocminfo` to verify GPU is detected
- Verify `HSA_OVERRIDE_GFX_VERSION=10.3.0` is set
- Check that ROCm drivers are loaded

### If "HIP Error" during compilation:
- Verify hipcc path is correct
- Check that ROCm is properly installed

### If kernel fails to launch:
- Check GPU memory usage (`rocm-smi`)
- Verify GPU is not in use by other processes

### If results are wrong:
- The kernel adds 1.0 to each input element
- Input: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
- Expected output: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

## Success Criteria:
- ✅ No HIP errors during execution
- ✅ Output shows "TEST PASSED"
- ✅ GPU memory operations work
- ✅ Kernel execution succeeds

If this test passes, your HIP setup is working correctly and any issues with MLX are likely in the MLX code itself (like the JIT compilation we implemented).
