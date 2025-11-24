// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/gemms/rocblas_gemm.h"
#include "mlx/backend/rocm/rocm_utils.h"
#include "mlx/backend/rocm/kernel_utils.cuh"

#include <rocblas/rocblas.h>

namespace mlx::core::rocm {

namespace {

rocblas_operation get_transpose_op(bool transposed) {
  return transposed ? rocblas_operation_transpose : rocblas_operation_none;
}

rocblas_datatype get_rocblas_type(Dtype dtype) {
  switch (dtype) {
    case float32:
      return rocblas_datatype_f32_r;
    case float16:
      return rocblas_datatype_f16_r;
    case bfloat16:
      return rocblas_datatype_bf16_r;
    default:
      throw std::runtime_error("Unsupported dtype for rocBLAS GEMM");
  }
}

} // namespace

void rocblas_gemm(
    CommandEncoder& encoder,
    bool a_transposed,
    bool b_transposed,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int64_t lda,
    const array& b,
    int64_t ldb,
    float beta,
    array& c,
    int64_t ldc,
    int batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t c_batch_stride) {
  
  auto handle = encoder.device().rocblas_handle();
  
  // Set stream for rocBLAS
  CHECK_ROCBLAS_ERROR(rocblas_set_stream(handle, encoder.stream()));
  
  auto dtype = a.dtype();
  auto rocblas_dtype = get_rocblas_type(dtype);
  
  // rocBLAS uses column-major, so we swap A and B
  // C = A * B becomes C^T = B^T * A^T in column-major
  rocblas_operation op_a = get_transpose_op(!b_transposed);
  rocblas_operation op_b = get_transpose_op(!a_transposed);
  
  // Swap dimensions accordingly
  int ld_a = static_cast<int>(ldb);
  int ld_b = static_cast<int>(lda);
  int ld_c = static_cast<int>(ldc);
  
  if (batch_count <= 1) {
    // Single GEMM
    if (dtype == float32) {
      CHECK_ROCBLAS_ERROR(rocblas_sgemm(
          handle,
          op_a, op_b,
          N, M, K,
          &alpha,
          static_cast<const float*>(b.data_shared_ptr()->buffer.raw_ptr()), ld_a,
          static_cast<const float*>(a.data_shared_ptr()->buffer.raw_ptr()), ld_b,
          &beta,
          static_cast<float*>(c.data_shared_ptr()->buffer.raw_ptr()), ld_c));
    } else {
      // For half precision, use rocblas_gemm_ex
      CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(
          handle,
          op_a, op_b,
          N, M, K,
          &alpha,
          b.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_a,
          a.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_b,
          &beta,
          c.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_c,
          c.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_c,
          rocblas_datatype_f32_r,  // compute type
          rocblas_gemm_algo_standard,
          0, 0));
    }
  } else {
    // Batched GEMM
    if (dtype == float32) {
      CHECK_ROCBLAS_ERROR(rocblas_sgemm_strided_batched(
          handle,
          op_a, op_b,
          N, M, K,
          &alpha,
          static_cast<const float*>(b.data_shared_ptr()->buffer.raw_ptr()), ld_a, b_batch_stride,
          static_cast<const float*>(a.data_shared_ptr()->buffer.raw_ptr()), ld_b, a_batch_stride,
          &beta,
          static_cast<float*>(c.data_shared_ptr()->buffer.raw_ptr()), ld_c, c_batch_stride,
          batch_count));
    } else {
      CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched_ex(
          handle,
          op_a, op_b,
          N, M, K,
          &alpha,
          b.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_a, b_batch_stride,
          a.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_b, a_batch_stride,
          &beta,
          c.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_c, c_batch_stride,
          c.data_shared_ptr()->buffer.raw_ptr(), rocblas_dtype, ld_c, c_batch_stride,
          batch_count,
          rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard,
          0, 0));
    }
  }
}

} // namespace mlx::core::rocm

