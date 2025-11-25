// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/utils.cuh"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/backend/common/reduce.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Use TypeConvert from kernel_utils.cuh for type conversions
// Alias for backwards compatibility within reduce code
template <typename T>
using ReduceTraits = TypeConvert<T>;

// Reduce operation types
template <typename T>
struct ReduceAdd {
  __device__ T operator()(T a, T b) { return a + b; }
  __device__ T init() { return ReduceTraits<T>::from_float(0.0f); }
};

template <typename T>
struct ReduceMul {
  __device__ T operator()(T a, T b) { return a * b; }
  __device__ T init() { return ReduceTraits<T>::from_float(1.0f); }
};

// Specialization for hip_bfloat16 comparison (convert to float for comparison)
template <typename T>
struct ReduceMax {
  __device__ T operator()(T a, T b) { return a > b ? a : b; }
  __device__ T init() { return Limits<T>::min(); }
};

template <>
struct ReduceMax<hip_bfloat16> {
  __device__ hip_bfloat16 operator()(hip_bfloat16 a, hip_bfloat16 b) {
    return static_cast<float>(a) > static_cast<float>(b) ? a : b;
  }
  __device__ hip_bfloat16 init() { return hip_bfloat16(-HUGE_VALF); }
};

template <>
struct ReduceMax<__half> {
  __device__ __half operator()(__half a, __half b) {
    return static_cast<float>(a) > static_cast<float>(b) ? a : b;
  }
  __device__ __half init() { return __half(-HUGE_VALF); }
};

template <typename T>
struct ReduceMin {
  __device__ T operator()(T a, T b) { return a < b ? a : b; }
  __device__ T init() { return Limits<T>::max(); }
};

template <>
struct ReduceMin<hip_bfloat16> {
  __device__ hip_bfloat16 operator()(hip_bfloat16 a, hip_bfloat16 b) {
    return static_cast<float>(a) < static_cast<float>(b) ? a : b;
  }
  __device__ hip_bfloat16 init() { return hip_bfloat16(HUGE_VALF); }
};

template <>
struct ReduceMin<__half> {
  __device__ __half operator()(__half a, __half b) {
    return static_cast<float>(a) < static_cast<float>(b) ? a : b;
  }
  __device__ __half init() { return __half(HUGE_VALF); }
};

// And/Or only make sense for bool, but we need to handle other types at compile time
template <typename T>
struct ReduceAnd {
  __device__ T operator()(T a, T b) { 
    return ReduceTraits<T>::from_float(
        (ReduceTraits<T>::to_float(a) != 0.0f && ReduceTraits<T>::to_float(b) != 0.0f) ? 1.0f : 0.0f);
  }
  __device__ T init() { return ReduceTraits<T>::from_float(1.0f); }
};

template <>
struct ReduceAnd<bool> {
  __device__ bool operator()(bool a, bool b) { return a && b; }
  __device__ bool init() { return true; }
};

template <typename T>
struct ReduceOr {
  __device__ T operator()(T a, T b) {
    return ReduceTraits<T>::from_float(
        (ReduceTraits<T>::to_float(a) != 0.0f || ReduceTraits<T>::to_float(b) != 0.0f) ? 1.0f : 0.0f);
  }
  __device__ T init() { return ReduceTraits<T>::from_float(0.0f); }
};

template <>
struct ReduceOr<bool> {
  __device__ bool operator()(bool a, bool b) { return a || b; }
  __device__ bool init() { return false; }
};

// Custom atomic operations for types without native atomicAdd support
// Uses compare-and-swap for generic implementation

template <typename T, typename Op>
__device__ void atomic_reduce(T* addr, T val, Op op) {
  if constexpr (sizeof(T) == 1) {
    // For 1-byte types, use 32-bit CAS with byte masking
    size_t addr_int = reinterpret_cast<size_t>(addr);
    int byte_offset = addr_int & 3;
    unsigned int* addr32 = reinterpret_cast<unsigned int*>(addr_int & ~3ULL);
    unsigned int old = *addr32;
    unsigned int assumed;
    do {
      assumed = old;
      unsigned char old_byte = (assumed >> (byte_offset * 8)) & 0xFF;
      T old_val;
      memcpy(&old_val, &old_byte, 1);
      T new_val = op(old_val, val);
      unsigned char new_byte;
      memcpy(&new_byte, &new_val, 1);
      unsigned int mask = 0xFFU << (byte_offset * 8);
      unsigned int new32 = (assumed & ~mask) | (static_cast<unsigned int>(new_byte) << (byte_offset * 8));
      old = atomicCAS(addr32, assumed, new32);
    } while (assumed != old);
  } else if constexpr (sizeof(T) == 2) {
    // For 16-bit types, use 32-bit CAS on aligned address
    size_t addr_int = reinterpret_cast<size_t>(addr);
    bool is_high = (addr_int & 2) != 0;
    unsigned int* addr32 = reinterpret_cast<unsigned int*>(addr_int & ~3ULL);
    unsigned int old = *addr32;
    unsigned int assumed;
    do {
      assumed = old;
      unsigned short old_val16 = is_high ? (assumed >> 16) : (assumed & 0xFFFF);
      T old_val;
      memcpy(&old_val, &old_val16, sizeof(T));
      T new_val = op(old_val, val);
      unsigned short new_val16;
      memcpy(&new_val16, &new_val, sizeof(T));
      unsigned int new32 = is_high ? 
          ((assumed & 0xFFFF) | (static_cast<unsigned int>(new_val16) << 16)) :
          ((assumed & 0xFFFF0000) | new_val16);
      old = atomicCAS(addr32, assumed, new32);
    } while (assumed != old);
  } else if constexpr (sizeof(T) == 4) {
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = *addr_as_uint;
    unsigned int assumed;
    do {
      assumed = old;
      T old_val;
      memcpy(&old_val, &assumed, sizeof(T));
      T new_val = op(old_val, val);
      unsigned int new_uint;
      memcpy(&new_uint, &new_val, sizeof(T));
      old = atomicCAS(addr_as_uint, assumed, new_uint);
    } while (assumed != old);
  } else if constexpr (sizeof(T) == 8) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    do {
      assumed = old;
      T old_val;
      memcpy(&old_val, &assumed, sizeof(T));
      T new_val = op(old_val, val);
      unsigned long long new_ull;
      memcpy(&new_ull, &new_val, sizeof(T));
      old = atomicCAS(addr_as_ull, assumed, new_ull);
    } while (assumed != old);
  }
}

// Trait to determine if native atomicAdd is available
template <typename T>
struct has_native_atomic_add : std::false_type {};

template <>
struct has_native_atomic_add<float> : std::true_type {};
template <>
struct has_native_atomic_add<int32_t> : std::true_type {};
template <>
struct has_native_atomic_add<uint32_t> : std::true_type {};
template <>
struct has_native_atomic_add<unsigned long long> : std::true_type {};

// Warp-level reduction
template <typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    // __shfl_down may not preserve bf16/half types - use bit casting
    if constexpr (sizeof(T) == 2) {
      unsigned short val_bits;
      memcpy(&val_bits, &val, sizeof(T));
      unsigned short shuffled_bits = __shfl_down(val_bits, offset);
      T shuffled_val;
      memcpy(&shuffled_val, &shuffled_bits, sizeof(T));
      val = op(val, shuffled_val);
    } else if constexpr (sizeof(T) == 1) {
      unsigned int val_int;
      unsigned char val_byte;
      memcpy(&val_byte, &val, 1);
      val_int = val_byte;
      unsigned int shuffled_int = __shfl_down(val_int, offset);
      unsigned char shuffled_byte = static_cast<unsigned char>(shuffled_int);
      T shuffled_val;
      memcpy(&shuffled_val, &shuffled_byte, 1);
      val = op(val, shuffled_val);
    } else {
      val = op(val, __shfl_down(val, offset));
    }
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
    // Use native atomicAdd for supported types, custom CAS-based for others
    if constexpr (has_native_atomic_add<T>::value) {
      atomicAdd(out, acc);
    } else {
      atomic_reduce(out, acc, op);
    }
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

