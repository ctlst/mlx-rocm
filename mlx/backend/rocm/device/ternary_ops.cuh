// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

struct Select {
  template <typename T>
  __device__ T operator()(bool cond, T a, T b) {
    return cond ? a : b;
  }
};

} // namespace mlx::core::rocm

