// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/device/utils.cuh"
#include "mlx/dtype_utils.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

// Alias TypeTag to type_identity for compatibility
template <typename T>
using TypeTag = type_identity<T>;

// Type mapping from MLX types to HIP types
template <typename T>
struct hip_type {
  using type = T;
};

template <>
struct hip_type<float16_t> {
  using type = __half;
};

template <>
struct hip_type<bfloat16_t> {
  using type = hip_bfloat16;
};

template <typename T>
using hip_type_t = typename hip_type<T>::type;

// Get GPU pointer from array
template <typename T = void>
inline T* gpu_ptr(const array& arr) {
  return static_cast<T*>(arr.data_shared_ptr()->buffer.raw_ptr());
}

// Block dimension helper
inline dim3 get_block_dims(int dim0, int dim1, int dim2) {
  int pow2 = 1;
  while (pow2 < dim0) {
    pow2 *= 2;
  }
  dim0 = std::min(pow2, 256);
  dim1 = std::min(dim1, 256 / dim0);
  dim2 = std::min(dim2, 256 / (dim0 * dim1));
  return dim3(dim0, dim1, dim2);
}

// Launch configuration helper
inline std::pair<dim3, dim3> get_launch_args(
    size_t size,
    const mlx::core::Shape& shape,
    const mlx::core::Strides& strides,
    bool large,
    int n_reads = 1) {
  int block_size = 256;
  size_t n_blocks = (size + block_size * n_reads - 1) / (block_size * n_reads);
  return {dim3(n_blocks), dim3(block_size)};
}

// Note: dispatch_all_types is provided by mlx/dtype_utils.h
// Use TypeTag<T> (alias for type_identity<T>) with lambdas

// Dispatch helper for bool
template <typename F>
void dispatch_bool(bool value, F&& f) {
  if (value) {
    f(std::true_type{});
  } else {
    f(std::false_type{});
  }
}

// Const parameter helper
template <typename T>
struct ConstParam {
  T data[MAX_NDIM];
  
  ConstParam() = default;
  
  template <typename Container>
  ConstParam(const Container& c) {
    size_t i = 0;
    for (const auto& v : c) {
      if (i >= MAX_NDIM) break;
      data[i++] = static_cast<T>(v);
    }
    while (i < MAX_NDIM) {
      data[i++] = 0;
    }
  }
  
  __host__ __device__ const T* operator()() const {
    return data;
  }
};

template <typename Container>
ConstParam<int32_t> const_param(const Container& shape) {
  return ConstParam<int32_t>(shape);
}

template <typename Container>
ConstParam<int64_t> const_param_strides(const Container& strides) {
  return ConstParam<int64_t>(strides);
}

// Collapse contiguous dimensions for optimization
inline std::pair<std::vector<int>, std::vector<int64_t>> collapse_contiguous_dims(
    const array& arr) {
  std::vector<int> shape;
  std::vector<int64_t> strides;
  
  if (arr.ndim() == 0) {
    return {shape, strides};
  }
  
  shape.push_back(arr.shape(0));
  strides.push_back(arr.strides(0));
  
  for (int i = 1; i < arr.ndim(); ++i) {
    if (strides.back() == arr.strides(i) * arr.shape(i)) {
      shape.back() *= arr.shape(i);
      strides.back() = arr.strides(i);
    } else {
      shape.push_back(arr.shape(i));
      strides.push_back(arr.strides(i));
    }
  }
  
  return {shape, strides};
}

} // namespace mlx::core::rocm

