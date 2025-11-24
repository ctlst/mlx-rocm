// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <roctracer/roctx.h>
#include <numeric>

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array>
check_transpose(rocm::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("Matmul::eval_gpu");
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  
  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    roctxRangePop();
    return;
  }

  out.set_data(rocm::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  // Calculate batch info
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);
  auto batch_count = out.size() / (M * N);
  
  int64_t a_batch_stride = batch_count > 1 ? a_batch_strides.back() : 0;
  int64_t b_batch_stride = batch_count > 1 ? b_batch_strides.back() : 0;
  int64_t c_batch_stride = batch_count > 1 ? M * N : 0;

  rocm::rocblas_gemm(
      encoder,
      a_transposed,
      b_transposed,
      M, N, K,
      1.0f,
      a, lda,
      b, ldb,
      0.0f,
      out, N,
      batch_count,
      a_batch_stride,
      b_batch_stride,
      c_batch_stride);
  
  roctxRangePop();
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("AddMM::eval_gpu");
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  // Return c if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    copy_gpu(c, out, c.flags().contiguous ? CopyType::Vector : CopyType::General, s);
    roctxRangePop();
    return;
  }

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  // Copy c to output first
  copy_gpu(c, out, c.flags().contiguous ? CopyType::Vector : CopyType::General, s);

  // Calculate batch info
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);
  auto batch_count = out.size() / (M * N);
  
  int64_t a_batch_stride = batch_count > 1 ? a_batch_strides.back() : 0;
  int64_t b_batch_stride = batch_count > 1 ? b_batch_strides.back() : 0;
  int64_t c_batch_stride = batch_count > 1 ? M * N : 0;

  rocm::rocblas_gemm(
      encoder,
      a_transposed,
      b_transposed,
      M, N, K,
      alpha_,
      a, lda,
      b, ldb,
      beta_,
      out, N,
      batch_count,
      a_batch_stride,
      b_batch_stride,
      c_batch_stride);
  
  roctxRangePop();
}

} // namespace mlx::core

