// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/utils.cuh"

#include <hip/hip_runtime.h>
#include <cmath>

namespace mlx::core::rocm {

// Helper to detect floating-point-like types (including hip_bfloat16 and __half)
template <typename T>
struct is_floating_like : std::is_floating_point<T> {};

template <>
struct is_floating_like<hip_bfloat16> : std::true_type {};

template <>
struct is_floating_like<__half> : std::true_type {};

template <typename T>
inline constexpr bool is_floating_like_v = is_floating_like<T>::value;

// Helper to promote types for binary operations
// Priority: double > float > hip_bfloat16/__half > int64 > int32 > smaller ints > bool
template <typename T, typename U>
struct BinaryResultType {
  // Default: use the larger type, with T as fallback
  using type = std::conditional_t<(sizeof(T) >= sizeof(U)), T, U>;
};

// Specialize for bool - always promote to the other type
template <typename U>
struct BinaryResultType<bool, U> {
  using type = U;
};

template <typename T>
struct BinaryResultType<T, bool> {
  using type = T;
};

template <>
struct BinaryResultType<bool, bool> {
  using type = bool;
};

template <typename T, typename U>
using common_type_t = typename BinaryResultType<T, U>::type;

struct Add {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    return static_cast<common_type_t<T, U>>(a) + static_cast<common_type_t<T, U>>(b);
  }
};

struct Subtract {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    return static_cast<common_type_t<T, U>>(a) - static_cast<common_type_t<T, U>>(b);
  }
};

struct Multiply {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    return static_cast<common_type_t<T, U>>(a) * static_cast<common_type_t<T, U>>(b);
  }
};

struct Divide {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    return static_cast<common_type_t<T, U>>(a) / static_cast<common_type_t<T, U>>(b);
  }
};

struct Remainder {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    if constexpr (std::is_integral_v<R>) {
      return static_cast<R>(a) % static_cast<R>(b);
    } else {
      return static_cast<R>(fmod(static_cast<double>(a), static_cast<double>(b)));
    }
  }
};

struct Maximum {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    if constexpr (is_floating_like_v<R>) {
      return static_cast<R>(fmax(static_cast<double>(a), static_cast<double>(b)));
    } else {
      return a > b ? static_cast<R>(a) : static_cast<R>(b);
    }
  }
};

struct Minimum {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    if constexpr (is_floating_like_v<R>) {
      return static_cast<R>(fmin(static_cast<double>(a), static_cast<double>(b)));
    } else {
      return a < b ? static_cast<R>(a) : static_cast<R>(b);
    }
  }
};

struct Power {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(pow(static_cast<double>(a), static_cast<double>(b)));
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
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};

struct LogicalOr {
  template <typename T, typename U>
  __device__ bool operator()(T a, U b) {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};

struct BitwiseAnd {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) & static_cast<R>(b);
  }
};

struct BitwiseOr {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) | static_cast<R>(b);
  }
};

struct BitwiseXor {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) ^ static_cast<R>(b);
  }
};

struct LeftShift {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) << static_cast<R>(b);
  }
};

struct RightShift {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(a) >> static_cast<R>(b);
  }
};

struct ArcTan2 {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(atan2(static_cast<double>(a), static_cast<double>(b)));
  }
};

struct LogAddExp {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(T a, U b) {
    using R = common_type_t<T, U>;
    double ra = static_cast<double>(a);
    double rb = static_cast<double>(b);
    double max_val = fmax(ra, rb);
    double min_val = fmin(ra, rb);
    return static_cast<R>(max_val + log1p(exp(min_val - max_val)));
  }
};

} // namespace mlx::core::rocm

