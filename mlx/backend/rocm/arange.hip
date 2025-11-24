// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <roctracer/roctx.h>

namespace mlx::core {

namespace rocm {

template <typename T>
__global__ void arange_kernel(T* out, T start, T step, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = start + static_cast<T>(idx) * step;
  }
}

} // namespace rocm

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("Arange::eval_gpu");
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  
  size_t size = out.size();
  if (size == 0) {
    roctxRangePop();
    return;
  }
  
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  dispatch_all_types(out.dtype(), [&](auto type_tag) {
    using T = rocm::hip_type_t<typename decltype(type_tag)::type>;
    T start_val = static_cast<T>(start_);
    T step_val = static_cast<T>(step_);
    
    hipLaunchKernelGGL(
        rocm::arange_kernel<T>,
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        rocm::gpu_ptr<T>(out),
        start_val,
        step_val,
        size);
  });
  
  roctxRangePop();
}

} // namespace mlx::core

