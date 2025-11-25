// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.cuh"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"

namespace mlx::core::rocm {

template <typename T>
void copy_contiguous_impl(
    CommandEncoder& encoder,
    const array& src,
    array& dst) {
  using HipT = hip_type_t<T>;
  
  auto size = src.data_size();
  if (size == 0) {
    return;
  }
  
  constexpr int N_READS = 4;
  int block_size = 256;
  int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
  
  if (size > UINT32_MAX) {
    // Store all arguments in local variables so we can take their addresses
    auto* src_ptr = gpu_ptr<HipT>(src);
    auto* dst_ptr = gpu_ptr<HipT>(dst);
    int64_t size_val = static_cast<int64_t>(size);
    rocm::rocm_launch_kernel(
        copy_contiguous_v_kernel<HipT, int64_t, N_READS>,
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        src_ptr, dst_ptr, size_val);
  } else {
    // Store all arguments in local variables so we can take their addresses
    auto* src_ptr = gpu_ptr<HipT>(src);
    auto* dst_ptr = gpu_ptr<HipT>(dst);
    uint32_t size_val = static_cast<uint32_t>(size);
    rocm::rocm_launch_kernel(
        copy_contiguous_v_kernel<HipT, uint32_t, N_READS>,
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        src_ptr, dst_ptr, size_val);
  }
}

void copy_contiguous(
    CommandEncoder& encoder,
    const array& src,
    array& dst) {
  // Use ROCm-specific dispatch that excludes complex64
  dispatch_all_types_rocm(src.dtype(), [&](auto type_tag) {
    using T = typename decltype(type_tag)::type;
    copy_contiguous_impl<T>(encoder, src, dst);
  });
}

} // namespace mlx::core::rocm

