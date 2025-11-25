// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/binary/binary.cuh"

namespace mlx::core {
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("Equal::eval_gpu");
  auto& s = out.primitive().stream();
  if (equal_nan_) {
    binary_op_gpu<rocm::NaNEqual>(inputs, out, name(), s);
  } else {
    binary_op_gpu<rocm::Equal>(inputs, out, name(), s);
  }
  roctxRangePop();
}
} // namespace mlx::core

