// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/reduce/reduce.cuh"

namespace mlx::core::rocm {

void init_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type) {
  out.set_data(malloc_async(out.nbytes(), encoder));
  
  size_t size = out.size();
  if (size == 0) return;
  
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  // Use ROCm-specific dispatch that excludes complex64
  dispatch_numeric_types_rocm(out.dtype(), [&](auto type_tag) {
    using T = hip_type_t<typename decltype(type_tag)::type>;
    dispatch_reduce_op<T>(type, [&](auto op) {
      auto* out_ptr = gpu_ptr<T>(out);
      size_t size_val = size;
      auto op_val = op;
      rocm::rocm_launch_kernel(
          init_reduce_kernel<T, decltype(op)>,
          dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
          out_ptr, size_val, op_val);
    });
  });
}

} // namespace mlx::core::rocm

