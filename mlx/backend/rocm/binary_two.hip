// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
// roctracer is optional - only used for profiling markers
#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#else
#define roctxRangePush(x) ((void)0)
#define roctxRangePop() ((void)0)
#endif

namespace mlx::core {

// BitwiseBinary combines multiple bitwise operations
void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("BitwiseBinary::eval_gpu");
  throw std::runtime_error("BitwiseBinary not yet implemented for ROCm");
  roctxRangePop();
}

// DivMod returns both quotient and remainder
void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  roctxRangePush("DivMod::eval_gpu");
  throw std::runtime_error("DivMod not yet implemented for ROCm");
  roctxRangePop();
}

} // namespace mlx::core

