// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/ternary_ops.cuh"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
// roctracer is optional - only used for profiling markers
#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#else
#define roctxRangePush(x) ((void)0)
#define roctxRangePop() ((void)0)
#endif

namespace mlx::core {

namespace rocm {

template <typename T, typename IdxT>
__global__ void select_kernel(
    const bool* __restrict__ cond,
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    IdxT size) {
  IdxT idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = cond[idx] ? a[idx] : b[idx];
  }
}

} // namespace rocm

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("Select::eval_gpu");
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  auto& cond = inputs[0];
  auto& a = inputs[1];
  auto& b = inputs[2];
  
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  
  size_t size = out.size();
  if (size == 0) {
    roctxRangePop();
    return;
  }
  
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  // Use ROCm-specific dispatch that excludes complex64
  rocm::dispatch_all_types_rocm(out.dtype(), [&](auto type_tag) {
    using T = rocm::hip_type_t<typename decltype(type_tag)::type>;
    
    if (size > UINT32_MAX) {
      // Store all arguments in local variables so we can take their addresses
      auto* cond_ptr = rocm::gpu_ptr<bool>(cond);
      auto* a_ptr = rocm::gpu_ptr<T>(a);
      auto* b_ptr = rocm::gpu_ptr<T>(b);
      auto* out_ptr = rocm::gpu_ptr<T>(out);
      int64_t size_val = static_cast<int64_t>(size);
      rocm::rocm_launch_kernel(
          rocm::select_kernel<T, int64_t>,
          dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
          cond_ptr, a_ptr, b_ptr, out_ptr, size_val);
    } else {
      // Store all arguments in local variables so we can take their addresses
      auto* cond_ptr = rocm::gpu_ptr<bool>(cond);
      auto* a_ptr = rocm::gpu_ptr<T>(a);
      auto* b_ptr = rocm::gpu_ptr<T>(b);
      auto* out_ptr = rocm::gpu_ptr<T>(out);
      uint32_t size_val = static_cast<uint32_t>(size);
      rocm::rocm_launch_kernel(
          rocm::select_kernel<T, uint32_t>,
          dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
          cond_ptr, a_ptr, b_ptr, out_ptr, size_val);
    }
  });
  
  roctxRangePop();
}

} // namespace mlx::core

