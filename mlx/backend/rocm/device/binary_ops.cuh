// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/utils.cuh"

#include <hip/hip_runtime.h>
#include <cmath>

namespace mlx::core::rocm {

// Helper to get common type for binary operations
template <typename T, typename U>
using common_type_t = decltype(T{} + U{});

struct Add {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    return a + b;
  }
};

struct Subtract {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    return a - b;
  }
};

struct Multiply {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    return a * b;
  }
};

struct Divide {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    return a / b;
  }
};

struct Remainder {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    if constexpr (std::is_integral_v<R>) {
      return static_cast<R>(a) % static_cast<R>(b);
    } else {
      return fmod(static_cast<R>(a), static_cast<R>(b));
    }
  }
};

struct Maximum {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    if constexpr (std::is_floating_point_v<R>) {
      return fmax(static_cast<R>(a), static_cast<R>(b));
    } else {
      return a > b ? static_cast<R>(a) : static_cast<R>(b);
    }
  }
};

struct Minimum {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    if constexpr (std::is_floating_point_v<R>) {
      return fmin(static_cast<R>(a), static_cast<R>(b));
    } else {
      return a < b ? static_cast<R>(a) : static_cast<R>(b);
    }
  }
};

struct Power {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    return pow(static_cast<double>(a), static_cast<double>(b));
  }
};

struct Equal {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return a == b;
  }
};

struct NotEqual {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return a != b;
  }
};

struct Greater {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return a > b;
  }
};

struct GreaterEqual {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return a >= b;
  }
};

struct Less {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return a < b;
  }
};

struct LessEqual {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
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
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) & static_cast<R>(b);
  }
};

struct BitwiseOr {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) | static_cast<R>(b);
  }
};

struct BitwiseXor {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) ^ static_cast<R>(b);
  }
};

struct LeftShift {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) << static_cast<R>(b);
  }
};

struct RightShift {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) >> static_cast<R>(b);
  }
};

struct ArcTan2 {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    return atan2(static_cast<double>(a), static_cast<double>(b));
  }
};

struct LogAddExp {
  template <typename T, typename U>
  __device__ auto operator()(T a, U b) -> common_type_t<T, U> {
    using R = common_type_t<T, U>;
    R ra = static_cast<R>(a);
    R rb = static_cast<R>(b);
    R max_val = Maximum{}(ra, rb);
    R min_val = Minimum{}(ra, rb);
    return max_val + log1p(exp(min_val - max_val));
  }
};

} // namespace mlx::core::rocm

