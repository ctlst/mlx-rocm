// Copyright © 2025 Apple Inc.

// This file must not include any host-only code, utilities that work under both
// host and device can be put here.

#pragma once

#include "mlx/backend/rocm/device/complex.cuh"
#include "mlx/backend/rocm/device/config.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

///////////////////////////////////////////////////////////////////////////////
// HIP kernel utils
///////////////////////////////////////////////////////////////////////////////

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
using Shape = int32_t[MAX_NDIM];
using Strides = int64_t[MAX_NDIM];

// Vectorized load/store.
template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
  T val[N];

  __device__ T& operator[](int i) {
    return val[i];
  }

  __device__ T operator[](int i) const {
    return val[i];
  }
};

template <int N, typename T>
inline __host__ __device__ bool is_aligned(T* x) {
  return (reinterpret_cast<uintptr_t>(x) % (N * sizeof(T))) == 0;
}

template <int N, typename T>
inline __device__ AlignedVector<T, N> unsafe_load_vector(
    const T* ptr,
    uint32_t offset) {
  auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
  return from[offset];
}

template <int N, typename T>
inline __device__ AlignedVector<T, N> load_vector(
    const T* ptr,
    uint32_t offset) {
  if (is_aligned<N>(ptr)) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  } else {
    AlignedVector<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] = ptr[offset * N + i];
    }
    return v;
  }
}

template <int N, typename T, typename SizeT>
inline __device__ AlignedVector<T, N>
load_vector(const T* ptr, uint32_t offset, SizeT size, T fallback) {
  if (is_aligned<N>(ptr) && (offset + 1) * N <= size) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  } else {
    AlignedVector<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] = (N * offset + i) < size ? ptr[offset * N + i] : fallback;
    }
    return v;
  }
}

template <int N, typename T, typename SizeT>
inline __device__ AlignedVector<T, N> load_vector(
    const T* ptr,
    uint32_t offset,
    SizeT size,
    int64_t stride,
    T fallback) {
  if (is_aligned<N>(ptr) && stride == 1 && (offset + 1) * N <= size) {
    auto* from = reinterpret_cast<const AlignedVector<T, N>*>(ptr);
    return from[offset];
  } else {
    AlignedVector<T, N> v;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      v[i] =
          (N * offset + i) < size ? ptr[stride * (offset * N + i)] : fallback;
    }
    return v;
  }
}

template <int N, typename T>
inline __device__ void
unsafe_store_vector(T* ptr, uint32_t offset, const AlignedVector<T, N>& vec) {
  auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
  to[offset] = vec;
}

template <int N, typename T>
inline __device__ void
store_vector(T* ptr, uint32_t offset, const AlignedVector<T, N>& vec) {
  if (is_aligned<N>(ptr)) {
    auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
    to[offset] = vec;
  } else {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      ptr[offset * N + i] = vec[i];
    }
  }
}

template <int N, typename T, typename SizeT>
inline __device__ void store_vector(
    T* ptr,
    uint32_t offset,
    const AlignedVector<T, N>& vec,
    SizeT size) {
  if (is_aligned<N>(ptr) && (offset + 1) * N <= size) {
    auto* to = reinterpret_cast<AlignedVector<T, N>*>(ptr);
    to[offset] = vec;
  } else {
    for (int i = 0; (offset * N + i) < size && i < N; ++i) {
      ptr[offset * N + i] = vec[i];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Limits {
  static constexpr __host__ __device__ T max();
  static constexpr __host__ __device__ T min();
  static constexpr __host__ __device__ T finite_max();
  static constexpr __host__ __device__ T finite_min();
};

template <>
struct Limits<float> {
  static constexpr __host__ __device__ float max() {
    return __FLT_MAX__;
  }
  static constexpr __host__ __device__ float min() {
    return -__FLT_MAX__;
  }
  static constexpr __host__ __device__ float finite_max() {
    return __FLT_MAX__;
  }
  static constexpr __host__ __device__ float finite_min() {
    return -__FLT_MAX__;
  }
};

template <>
struct Limits<double> {
  static constexpr __host__ __device__ double max() {
    return __DBL_MAX__;
  }
  static constexpr __host__ __device__ double min() {
    return -__DBL_MAX__;
  }
  static constexpr __host__ __device__ double finite_max() {
    return __DBL_MAX__;
  }
  static constexpr __host__ __device__ double finite_min() {
    return -__DBL_MAX__;
  }
};

template <>
struct Limits<int32_t> {
  static constexpr __host__ __device__ int32_t max() {
    return INT32_MAX;
  }
  static constexpr __host__ __device__ int32_t min() {
    return INT32_MIN;
  }
  static constexpr __host__ __device__ int32_t finite_max() {
    return INT32_MAX;
  }
  static constexpr __host__ __device__ int32_t finite_min() {
    return INT32_MIN;
  }
};

template <>
struct Limits<int64_t> {
  static constexpr __host__ __device__ int64_t max() {
    return INT64_MAX;
  }
  static constexpr __host__ __device__ int64_t min() {
    return INT64_MIN;
  }
  static constexpr __host__ __device__ int64_t finite_max() {
    return INT64_MAX;
  }
  static constexpr __host__ __device__ int64_t finite_min() {
    return INT64_MIN;
  }
};

template <>
struct Limits<bool> {
  static constexpr __host__ __device__ bool max() {
    return true;
  }
  static constexpr __host__ __device__ bool min() {
    return false;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Optimize when the ndim is known at compile time.
template <int NDIM, typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc_nd(IdxT elem, const int* shape, const int64_t* strides) {
  IdxT loc = 0;
#pragma unroll
  for (int i = NDIM - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

} // namespace mlx::core::rocm

