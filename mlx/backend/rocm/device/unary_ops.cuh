// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/utils.cuh"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cmath>

namespace mlx::core::rocm {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x;
    } else {
      return abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return atan(x);
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return atanh(x);
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else {
      return ceil(x);
    }
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    return cos(x);
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    return cosh(x);
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return __half(erf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(erf(static_cast<float>(x)));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return __half(erfinv(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(erfinv(static_cast<float>(x)));
    } else {
      return erfinv(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    return exp(x);
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return __half(expm1(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(expm1(static_cast<float>(x)));
    } else {
      return expm1(x);
    }
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else {
      return floor(x);
    }
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return log(x);
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T x) {
    return log1p(x);
  }
};

struct LogicalNot {
  __device__ bool operator()(bool x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  __device__ T operator()(T x) {
    return -x;
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    return rint(x);
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = T(1) / (T(1) + exp(abs(x)));
    return (x < T(0)) ? y : T(1) - y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x != 0;
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    return sin(x);
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    return sinh(x);
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return sqrt(x);
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return rsqrt(x);
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    return tan(x);
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    return tanh(x);
  }
};

} // namespace mlx::core::rocm

