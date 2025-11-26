// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/unary/unary.cuh"

namespace mlx::core {

void Conjugate::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  throw std::runtime_error(
      "[ROCm] Conjugate operation not supported on GPU, use CPU fallback");
}

} // namespace mlx::core

