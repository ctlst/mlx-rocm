// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/binary/binary.cuh"

namespace mlx::core {
void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("BitwiseBinary::eval_gpu");
  auto& s = out.primitive().stream();
  switch (op_) {
    case BitwiseBinary::And:
      binary_op_gpu<rocm::BitwiseAnd>(inputs, out, name(), s);
      break;
    case BitwiseBinary::Or:
      binary_op_gpu<rocm::BitwiseOr>(inputs, out, name(), s);
      break;
    case BitwiseBinary::Xor:
      binary_op_gpu<rocm::BitwiseXor>(inputs, out, name(), s);
      break;
    case BitwiseBinary::LeftShift:
      binary_op_gpu<rocm::LeftShift>(inputs, out, name(), s);
      break;
    case BitwiseBinary::RightShift:
      binary_op_gpu<rocm::RightShift>(inputs, out, name(), s);
      break;
  }
  roctxRangePop();
}
} // namespace mlx::core

