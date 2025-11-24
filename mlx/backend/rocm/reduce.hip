// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/reduce/reduce.cuh"
#include "mlx/backend/gpu/copy.h"

#include <roctracer/roctx.h>

namespace mlx::core {

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("Reduce::eval_gpu");
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here.
  assert(!axes_.empty());
  assert(out.size() != in.size());

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  if (in.size() == 0) {
    rocm::init_reduce(encoder, in, out, reduce_type_);
    roctxRangePop();
    return;
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  bool broadcasted = false;
  for (int i = 0, j = 0; i < in.ndim() && !broadcasted; i++) {
    if (j < axes_.size() && axes_[j] == i) {
      j++;
    } else {
      broadcasted = in.strides(i) == 0;
    }
  }
  if (plan.type == GeneralReduce || broadcasted || !in.flags().contiguous) {
    array in_copy = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  if (plan.type == ContiguousAllReduce) {
    rocm::all_reduce(encoder, in, out, reduce_type_);
    roctxRangePop();
    return;
  }

  if (plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
    rocm::row_reduce(encoder, in, out, reduce_type_, axes_, plan);
    roctxRangePop();
    return;
  }

  if (plan.type == ContiguousStridedReduce ||
      plan.type == GeneralStridedReduce) {
    rocm::col_reduce(encoder, in, out, reduce_type_, axes_, plan);
    roctxRangePop();
    return;
  }

  throw std::runtime_error("No plan reached in reduce.");
}

} // namespace mlx::core

