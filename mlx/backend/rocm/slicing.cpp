// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <roctracer/roctx.h>

namespace mlx::core {

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  roctxRangePush("concatenate_gpu");
  auto& encoder = rocm::get_command_encoder(s);
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  
  size_t offset = 0;
  for (const auto& in : inputs) {
    if (in.size() == 0) continue;
    
    auto [data_offset, out_strides] =
        prepare_slice(out, offset, axis);
    
    copy_gpu_inplace(
        in, out, in.shape(), in.strides(), out_strides,
        0, data_offset, CopyType::GeneralGeneral, s);
    
    offset += in.shape(axis);
  }
  roctxRangePop();
}

void slice_gpu(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides,
    const Stream& s) {
  roctxRangePush("slice_gpu");
  auto& encoder = rocm::get_command_encoder(s);
  
  auto [data_offset, in_strides] =
      prepare_slice(in, start_indices, strides);
  
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  
  copy_gpu_inplace(
      in, out, out.shape(), in_strides, out.strides(),
      data_offset, 0, CopyType::GeneralGeneral, s);
  
  roctxRangePop();
}

void pad_gpu(
    const array& in,
    const array& val,
    array& out,
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Stream& s) {
  roctxRangePush("pad_gpu");
  auto& encoder = rocm::get_command_encoder(s);
  
  // Fill output with padding value
  fill_gpu(val, out, s);
  
  // Copy input to output at the appropriate offset
  Shape start(out.ndim(), 0);
  for (size_t i = 0; i < axes.size(); ++i) {
    start[axes[i]] = low_pad_size[i];
  }
  
  auto [data_offset, out_strides] =
      prepare_slice(out, start, Shape(out.ndim(), 1));
  
  copy_gpu_inplace(
      in, out, in.shape(), in.strides(), out_strides,
      0, data_offset, CopyType::GeneralGeneral, s);
  
  roctxRangePop();
}

array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  // Simplified implementation - compute on CPU for now
  // TODO: Implement GPU kernel for dynamic offset computation
  throw std::runtime_error("Dynamic slice not yet implemented for ROCm");
}

} // namespace mlx::core

