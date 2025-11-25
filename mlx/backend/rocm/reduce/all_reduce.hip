// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/reduce/reduce.cuh"

namespace mlx::core::rocm {

void all_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type) {
  out.set_data(malloc_async(out.nbytes(), encoder));
  
  size_t size = in.size();
  if (size == 0) {
    init_reduce(encoder, in, out, type);
    return;
  }
  
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = std::min((size + BLOCK_SIZE - 1) / BLOCK_SIZE, static_cast<size_t>(256));
  
  // Use ROCm-specific dispatch that excludes complex64
  dispatch_numeric_types_rocm(in.dtype(), [&](auto type_tag) {
    using T = hip_type_t<typename decltype(type_tag)::type>;
    dispatch_reduce_op<T>(type, [&](auto op) {
      // Initialize output
      auto* out_ptr_init = gpu_ptr<T>(out);
      size_t init_size = static_cast<size_t>(1);
      auto op_init = op;
      rocm::rocm_launch_kernel(
          HIP_KERNEL_NAME(init_reduce_kernel<T, decltype(op)>),
          dim3(1), dim3(1), 0, encoder.stream(),
          out_ptr_init, init_size, op_init);

      // Perform reduction
      auto* in_ptr = gpu_ptr<T>(in);
      auto* out_ptr = gpu_ptr<T>(out);
      size_t reduce_size = size;
      auto op_reduce = op;
      rocm::rocm_launch_kernel(
          HIP_KERNEL_NAME(all_reduce_kernel<T, decltype(op), BLOCK_SIZE>),
          dim3(num_blocks), dim3(BLOCK_SIZE), 0, encoder.stream(),
          in_ptr, out_ptr, reduce_size, op_reduce);
    });
  });
}

} // namespace mlx::core::rocm

