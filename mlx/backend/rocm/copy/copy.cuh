// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/utils.cuh"
#include "mlx/backend/rocm/kernel_utils.cuh"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Contiguous copy kernel
template <typename T, typename IdxT>
__global__ void copy_contiguous_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    IdxT size) {
  IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = src[idx];
  }
}

// Vectorized contiguous copy kernel
template <typename T, typename IdxT, int N_READS>
__global__ void copy_contiguous_v_kernel(
    const T* __restrict__ src,
    T* __restrict__ dst,
    IdxT size) {
  IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT vec_idx = idx * N_READS;
  
  if ((idx + 1) * N_READS <= size) {
    auto vec = load_vector<N_READS>(src, idx);
    store_vector<N_READS>(dst, idx, vec);
  } else if (vec_idx < size) {
    for (IdxT i = vec_idx; i < size; ++i) {
      dst[i] = src[i];
    }
  }
}

// General copy kernel (handles non-contiguous arrays)
template <typename SrcT, typename DstT, typename IdxT>
__global__ void copy_general_kernel(
    const SrcT* __restrict__ src,
    DstT* __restrict__ dst,
    IdxT size,
    const int32_t* src_shape,
    const int64_t* src_strides,
    const int64_t* dst_strides,
    int ndim) {
  IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    IdxT src_loc = elem_to_loc(idx, src_shape, src_strides, ndim);
    IdxT dst_loc = elem_to_loc(idx, src_shape, dst_strides, ndim);
    dst[dst_loc] = static_cast<DstT>(src[src_loc]);
  }
}

// Scalar broadcast copy kernel
template <typename SrcT, typename DstT, typename IdxT>
__global__ void copy_scalar_kernel(
    const SrcT* __restrict__ src,
    DstT* __restrict__ dst,
    IdxT size) {
  IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dst[idx] = static_cast<DstT>(*src);
  }
}

} // namespace mlx::core::rocm

