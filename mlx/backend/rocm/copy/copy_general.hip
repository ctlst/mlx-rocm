// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.cuh"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"

namespace mlx::core::rocm {

template <typename SrcT, typename DstT>
void copy_general_impl(
    CommandEncoder& encoder,
    const array& src,
    array& dst,
    const mlx::core::Shape& data_shape,
    const mlx::core::Strides& src_strides,
    const mlx::core::Strides& dst_strides,
    int64_t src_offset,
    int64_t dst_offset) {
  using HipSrcT = hip_type_t<SrcT>;
  using HipDstT = hip_type_t<DstT>;
  
  auto size = dst.size();
  if (size == 0) {
    return;
  }
  
  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  
  // Copy shape and strides to device-accessible arrays
  ConstParam<int32_t> shape_param;
  ConstParam<int64_t> src_strides_param;
  ConstParam<int64_t> dst_strides_param;
  
  int ndim = dst.ndim();
  for (int i = 0; i < ndim && i < MAX_NDIM; ++i) {
    shape_param.data[i] = data_shape[i];
    src_strides_param.data[i] = src_strides[i];
    dst_strides_param.data[i] = dst_strides[i];
  }
  
  if (size > UINT32_MAX) {
    // Store all arguments in local variables so we can take their addresses
    auto* src_ptr = gpu_ptr<HipSrcT>(src) + src_offset;
    auto* dst_ptr = gpu_ptr<HipDstT>(dst) + dst_offset;
    int64_t size_val = static_cast<int64_t>(size);
    int32_t* shape_data = shape_param.data;
    int64_t* src_strides_data = src_strides_param.data;
    int64_t* dst_strides_data = dst_strides_param.data;
    rocm::rocm_launch_kernel(
        copy_general_kernel<HipSrcT, HipDstT, int64_t>,
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        src_ptr, dst_ptr, size_val, shape_data, src_strides_data, dst_strides_data, ndim);
  } else {
    // Store all arguments in local variables so we can take their addresses
    auto* src_ptr = gpu_ptr<HipSrcT>(src) + src_offset;
    auto* dst_ptr = gpu_ptr<HipDstT>(dst) + dst_offset;
    uint32_t size_val = static_cast<uint32_t>(size);
    int32_t* shape_data = shape_param.data;
    int64_t* src_strides_data = src_strides_param.data;
    int64_t* dst_strides_data = dst_strides_param.data;
    rocm::rocm_launch_kernel(
        copy_general_kernel<HipSrcT, HipDstT, uint32_t>,
        dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
        src_ptr, dst_ptr, size_val, shape_data, src_strides_data, dst_strides_data, ndim);
  }
}

void copy_general(
    CommandEncoder& encoder,
    const array& src,
    array& dst,
    const mlx::core::Shape& data_shape,
    const mlx::core::Strides& src_strides,
    const mlx::core::Strides& dst_strides,
    int64_t src_offset,
    int64_t dst_offset) {
  // Use ROCm-specific dispatch that excludes complex64
  dispatch_all_types_rocm(src.dtype(), [&](auto src_type_tag) {
    dispatch_all_types_rocm(dst.dtype(), [&](auto dst_type_tag) {
      using SrcT = typename decltype(src_type_tag)::type;
      using DstT = typename decltype(dst_type_tag)::type;
      copy_general_impl<SrcT, DstT>(
          encoder, src, dst, data_shape, src_strides, dst_strides,
          src_offset, dst_offset);
    });
  });
}

} // namespace mlx::core::rocm

