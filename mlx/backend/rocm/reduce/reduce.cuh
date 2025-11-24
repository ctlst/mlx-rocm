// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/utils.cuh"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/backend/common/reduce.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Reduce operation types
template <typename T>
struct ReduceAdd {
  __device__ T operator()(T a, T b) { return a + b; }
  __device__ T init() { return T(0); }
};

template <typename T>
struct ReduceMul {
  __device__ T operator()(T a, T b) { return a * b; }
  __device__ T init() { return T(1); }
};

template <typename T>
struct ReduceMax {
  __device__ T operator()(T a, T b) { return a > b ? a : b; }
  __device__ T init() { return Limits<T>::min(); }
};

template <typename T>
struct ReduceMin {
  __device__ T operator()(T a, T b) { return a < b ? a : b; }
  __device__ T init() { return Limits<T>::max(); }
};

template <typename T>
struct ReduceAnd {
  __device__ T operator()(T a, T b) { return a && b; }
  __device__ T init() { return T(true); }
};

template <typename T>
struct ReduceOr {
  __device__ T operator()(T a, T b) { return a || b; }
  __device__ T init() { return T(false); }
};

// Warp-level reduction
template <typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = op(val, __shfl_down(val, offset));
  }
  return val;
}

// Block-level reduction
template <typename T, typename Op, int BLOCK_SIZE>
__device__ T block_reduce(T val, Op op, T* shared) {
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;
  
  // First reduce within warp
  val = warp_reduce(val, op);
  
  // Write reduced value to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  
  // Read from shared memory and reduce
  constexpr int numWarps = BLOCK_SIZE / 64; // gfx1030 uses wave64
  val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : op.init();
  
  if (wid == 0) {
    val = warp_reduce(val, op);
  }
  
  return val;
}

// All-reduce kernel
template <typename T, typename Op, int BLOCK_SIZE>
__global__ void all_reduce_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    size_t size,
    Op op) {
  __shared__ T shared[BLOCK_SIZE / 64];
  
  T acc = op.init();
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  
  for (size_t i = idx; i < size; i += stride) {
    acc = op(acc, in[i]);
  }
  
  acc = block_reduce<T, Op, BLOCK_SIZE>(acc, op, shared);
  
  if (threadIdx.x == 0) {
    atomicAdd(out, acc);
  }
}

// Row reduce kernel (reduce along last axis)
template <typename T, typename Op, int BLOCK_SIZE>
__global__ void row_reduce_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int64_t reduction_size,
    int64_t out_size,
    Op op) {
  __shared__ T shared[BLOCK_SIZE / 64];
  
  int64_t out_idx = blockIdx.x;
  if (out_idx >= out_size) return;
  
  const T* in_row = in + out_idx * reduction_size;
  
  T acc = op.init();
  for (int64_t i = threadIdx.x; i < reduction_size; i += blockDim.x) {
    acc = op(acc, in_row[i]);
  }
  
  acc = block_reduce<T, Op, BLOCK_SIZE>(acc, op, shared);
  
  if (threadIdx.x == 0) {
    out[out_idx] = acc;
  }
}

// Col reduce kernel (reduce along non-last axes)
template <typename T, typename Op, int BLOCK_SIZE>
__global__ void col_reduce_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int64_t reduction_size,
    int64_t reduction_stride,
    int64_t out_size,
    Op op) {
  int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= out_size) return;
  
  const T* in_col = in + out_idx;
  
  T acc = op.init();
  for (int64_t i = 0; i < reduction_size; ++i) {
    acc = op(acc, in_col[i * reduction_stride]);
  }
  
  out[out_idx] = acc;
}

// Init reduce (set output to initial value)
template <typename T, typename Op>
__global__ void init_reduce_kernel(T* out, size_t size, Op op) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = op.init();
  }
}

// Helper to select reduce operation
template <typename T, typename F>
void dispatch_reduce_op(Reduce::ReduceType type, F&& f) {
  switch (type) {
    case Reduce::Sum:
      f(ReduceAdd<T>{});
      break;
    case Reduce::Prod:
      f(ReduceMul<T>{});
      break;
    case Reduce::Max:
      f(ReduceMax<T>{});
      break;
    case Reduce::Min:
      f(ReduceMin<T>{});
      break;
    case Reduce::And:
      f(ReduceAnd<T>{});
      break;
    case Reduce::Or:
      f(ReduceOr<T>{});
      break;
    default:
      throw std::runtime_error("Unsupported reduce type");
  }
}

void init_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type);

void all_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type);

void row_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

void col_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

} // namespace mlx::core::rocm

