// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/reduce/reduce.cuh"

namespace mlx::core::rocm {

void row_reduce(
    CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  out.set_data(malloc_async(out.nbytes(), encoder));
  
  int64_t reduction_size = plan.shape.back();
  int64_t out_size = out.size();
  
  if (out_size == 0) return;
  
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = out_size;
  
  // Use ROCm-specific dispatch that excludes complex64
  dispatch_numeric_types_rocm(in.dtype(), [&](auto type_tag) {
    using T = hip_type_t<typename decltype(type_tag)::type>;
    dispatch_reduce_op<T>(type, [&](auto op) {
      auto* in_ptr = gpu_ptr<T>(in);
      auto* out_ptr = gpu_ptr<T>(out);
      int64_t red_size = reduction_size;
      int64_t out_size_val = out_size;
      auto op_val = op;
      rocm::rocm_launch_kernel(
          row_reduce_kernel<T, decltype(op), BLOCK_SIZE>,
          dim3(num_blocks), dim3(BLOCK_SIZE), 0, encoder.stream(),
          in_ptr, out_ptr, red_size, out_size_val, op_val);
    });
  });
}

} // namespace mlx::core::rocm

