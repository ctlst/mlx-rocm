// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core::rocm {

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
    int batch_count = 1,
    int64_t a_batch_stride = 0,
    int64_t b_batch_stride = 0,
    int64_t c_batch_stride = 0);

} // namespace mlx::core::rocm

