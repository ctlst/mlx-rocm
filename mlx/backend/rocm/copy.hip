// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.cuh"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace rocm {

void copy_contiguous(CommandEncoder& encoder, const array& src, array& dst);
void copy_general(
    CommandEncoder& encoder,
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& src_strides,
    const Strides& dst_strides,
    int64_t src_offset,
    int64_t dst_offset);

} // namespace rocm

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  switch (ctype) {
    case CopyType::Scalar:
    case CopyType::Vector:
      rocm::copy_contiguous(encoder, in, out);
      break;
    case CopyType::General:
    case CopyType::GeneralGeneral:
      rocm::copy_general(
          encoder, in, out, data_shape, i_strides, o_strides,
          i_offset, o_offset);
      break;
  }
}

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s) {
  auto& encoder = rocm::get_command_encoder(s);
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  copy_gpu_inplace(
      src, out, out.shape(), src.strides(), out.strides(),
      0, 0, ctype, s);
}

void copy_gpu(const array& src, array& out, CopyType ctype) {
  copy_gpu(src, out, ctype, out.primitive().stream());
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  copy_gpu_inplace(
      in, out, out.shape(), in.strides(), out.strides(),
      0, 0, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Strides& i_strides,
    int64_t i_offset,
    CopyType ctype,
    const Stream& s) {
  copy_gpu_inplace(
      in, out, out.shape(), i_strides, out.strides(),
      i_offset, 0, ctype, s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  auto& encoder = rocm::get_command_encoder(s);
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  
  if (out.size() == 0) {
    return;
  }
  
  copy_gpu_inplace(
      val, out, out.shape(), Strides(out.ndim(), 0), out.strides(),
      0, 0, CopyType::General, s);
}

array contiguous_copy_gpu(const array& arr, const Stream& s) {
  array out(arr.shape(), arr.dtype(), nullptr, {});
  copy_gpu(arr, out, CopyType::General, s);
  return out;
}

void reshape_gpu(const array& in, array& out, Stream s) {
  auto& encoder = rocm::get_command_encoder(s);
  
  if (in.flags().row_contiguous) {
    out.copy_shared_buffer(in);
    return;
  }
  
  if (in.flags().col_contiguous) {
    // Need to transpose to row-major
    out.set_data(rocm::malloc_async(out.nbytes(), encoder));
    copy_gpu_inplace(in, out, CopyType::General, s);
    return;
  }
  
  // General case: need to copy
  out.set_data(rocm::malloc_async(out.nbytes(), encoder));
  copy_gpu_inplace(in, out, CopyType::General, s);
}

array flatten_in_eval(const array& x, int start_axis, int end_axis, Stream s) {
  // This is a simplified version
  auto shape = x.shape();
  int64_t new_dim = 1;
  for (int i = start_axis; i <= end_axis; ++i) {
    new_dim *= shape[i];
  }
  
  Shape new_shape;
  for (int i = 0; i < start_axis; ++i) {
    new_shape.push_back(shape[i]);
  }
  new_shape.push_back(new_dim);
  for (int i = end_axis + 1; i < x.ndim(); ++i) {
    new_shape.push_back(shape[i]);
  }
  
  array out(new_shape, x.dtype(), nullptr, {});
  reshape_gpu(x, out, s);
  return out;
}

array reshape_in_eval(const array& x, Shape shape, Stream s) {
  array out(shape, x.dtype(), nullptr, {});
  reshape_gpu(x, out, s);
  return out;
}

array swapaxes_in_eval(const array& x, int axis1, int axis2) {
  auto shape = x.shape();
  auto strides = x.strides();
  std::swap(shape[axis1], shape[axis2]);
  std::swap(strides[axis1], strides[axis2]);
  
  array out(shape, x.dtype(), nullptr, {});
  out.copy_shared_buffer(x, strides, x.flags(), x.data_size());
  return out;
}

} // namespace mlx::core

