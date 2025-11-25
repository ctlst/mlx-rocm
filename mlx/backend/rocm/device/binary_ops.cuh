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
  // When mixing floating-point and integer types, promote to floating-point
  using type = std::conditional_t<
    (is_floating_like_v<T> && std::is_integral_v<U>),
    std::conditional_t<std::is_same_v<T, double> || std::is_same_v<U, double>, double,
      std::conditional_t<std::is_same_v<T, float> || std::is_same_v<U, float> ||
                        std::is_same_v<T, hip_bfloat16> || std::is_same_v<U, hip_bfloat16> ||
                        std::is_same_v<T, __half> || std::is_same_v<U, __half>, float, T>>,
    std::conditional_t<
      (is_floating_like_v<U> && std::is_integral_v<T>),
      std::conditional_t<std::is_same_v<T, double> || std::is_same_v<U, double>, double,
        std::conditional_t<std::is_same_v<T, float> || std::is_same_v<U, float> ||
                          std::is_same_v<T, hip_bfloat16> || std::is_same_v<U, hip_bfloat16> ||
                          std::is_same_v<T, __half> || std::is_same_v<U, __half>, float, U>>,
      // Both same category: use larger size, prefer T if equal
      std::conditional_t<(sizeof(T) > sizeof(U)), T,
        std::conditional_t<(sizeof(U) > sizeof(T)), U, T>>
    >
  >;
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
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    return static_cast<CommonT>(a) == static_cast<CommonT>(b);
  }
};

struct NaNEqual {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    CommonT ca = static_cast<CommonT>(a);
    CommonT cb = static_cast<CommonT>(b);
    if constexpr (is_floating_like_v<CommonT>) {
      // For floating-point types, check NaN equality
      if constexpr (std::is_same_v<CommonT, float>) {
        return ca == cb || (__isnanf(ca) && __isnanf(cb));
      } else if constexpr (std::is_same_v<CommonT, double>) {
        return ca == cb || (__isnan(ca) && __isnan(cb));
      } else {
        // For other floating types, just do regular comparison
        return ca == cb;
      }
    } else {
      return ca == cb;
    }
  }
};

struct NotEqual {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    return static_cast<CommonT>(a) != static_cast<CommonT>(b);
  }
};

struct Greater {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    return static_cast<CommonT>(a) > static_cast<CommonT>(b);
  }
};

struct GreaterEqual {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    return static_cast<CommonT>(a) >= static_cast<CommonT>(b);
  }
};

struct Less {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    return static_cast<CommonT>(a) < static_cast<CommonT>(b);
  }
};

struct LessEqual {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    using CommonT = common_type_t<T, U>;
    return static_cast<CommonT>(a) <= static_cast<CommonT>(b);
  }
};

struct LogicalAnd {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    return static_cast<bool>(a) && static_cast<bool>(b);
  }
};

struct LogicalOr {
  template <typename T, typename U>
  __device__ bool operator()(const T& a, const U& b) {
    return static_cast<bool>(a) || static_cast<bool>(b);
  }
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x & y;
  }
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x | y;
  }
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x ^ y;
  }
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x << y;
  }
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x >> y;
  }
};

struct ArcTan2 {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(const T& a, const U& b) {
    using R = common_type_t<T, U>;
    return static_cast<R>(atan2(static_cast<double>(a), static_cast<double>(b)));
  }
};

struct LogAddExp {
  template <typename T, typename U>
  __device__ common_type_t<T, U> operator()(const T& a, const U& b) {
    using R = common_type_t<T, U>;
    double ra = static_cast<double>(a);
    double rb = static_cast<double>(b);
    double max_val = fmax(ra, rb);
    double min_val = fmin(ra, rb);
    return static_cast<R>(max_val + log1p(exp(min_val - max_val)));
  }
};

} // namespace mlx::core::rocm

