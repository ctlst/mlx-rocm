// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/utils.cuh"

#include <hip/hip_runtime.h>
#include <cmath>

namespace mlx::core::rocm {

struct Add {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a + b;
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a - b;
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a * b;
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a / b;
  }
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
      return a % b;
    } else {
      return fmod(a, b);
    }
  }
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
      return fmax(a, b);
    } else {
      return a > b ? a : b;
    }
  }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
      return fmin(a, b);
    } else {
      return a < b ? a : b;
    }
  }
};

struct Power {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return pow(a, b);
  }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a == b;
  }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a != b;
  }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a > b;
  }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a >= b;
  }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a < b;
  }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T a, T b) {
    return a <= b;
  }
};

struct LogicalAnd {
  __device__ bool operator()(bool a, bool b) {
    return a && b;
  }
};

struct LogicalOr {
  __device__ bool operator()(bool a, bool b) {
    return a || b;
  }
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a & b;
  }
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a | b;
  }
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a ^ b;
  }
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a << b;
  }
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a >> b;
  }
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return atan2(a, b);
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T a, T b) {
    T max_val = Maximum{}(a, b);
    T min_val = Minimum{}(a, b);
    return max_val + log1p(exp(min_val - max_val));
  }
};

} // namespace mlx::core::rocm

