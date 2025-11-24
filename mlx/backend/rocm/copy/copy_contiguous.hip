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
    hipLaunchKernelGGL(
        (copy_contiguous_v_kernel<HipT, int64_t, N_READS>),
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        gpu_ptr<HipT>(src),
        gpu_ptr<HipT>(dst),
        static_cast<int64_t>(size));
  } else {
    hipLaunchKernelGGL(
        (copy_contiguous_v_kernel<HipT, uint32_t, N_READS>),
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        gpu_ptr<HipT>(src),
        gpu_ptr<HipT>(dst),
        static_cast<uint32_t>(size));
  }
}

void copy_contiguous(
    CommandEncoder& encoder,
    const array& src,
    array& dst) {
  dispatch_all_types(src.dtype(), [&](auto type_tag) {
    using T = typename decltype(type_tag)::type;
    copy_contiguous_impl<T>(encoder, src, dst);
  });
}

} // namespace mlx::core::rocm

