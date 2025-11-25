// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

// roctracer is optional - only used for profiling markers
#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#define MLX_HAS_ROCTX 1
#else
#define MLX_HAS_ROCTX 0
#define roctxRangePush(x) ((void)0)
#define roctxRangePop() ((void)0)
#endif

namespace mlx::core::gpu {

bool is_available() {
  return true;
}

void new_stream(Stream s) {
  // Force initialization of HIP, so HIP runtime gets destroyed last.
  hipFree(nullptr);
  // Ensure the static stream objects get created.
  rocm::get_command_encoder(s);
}

void eval(array& arr) {
  roctxRangePush("gpu::eval");
  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  auto& stream = arr.primitive().stream();
  auto& encoder = rocm::get_command_encoder(stream);
  // Keep used buffers alive until kernel finishes running.
  for (auto& in : arr.inputs()) {
    // Except for the donated one.
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }

  if (encoder.needs_commit()) {
    scheduler::notify_new_task(stream);
    encoder.add_completed_handler(
        [stream]() { scheduler::notify_task_completion(stream); });
    encoder.commit();
  }
  roctxRangePop();
}

void finalize(Stream s) {
  roctxRangePush("gpu::finalize");
  rocm::get_command_encoder(s).commit();
  roctxRangePop();
}

void synchronize(Stream s) {
  roctxRangePush("gpu::synchronize");
  rocm::get_command_encoder(s).synchronize();
  roctxRangePop();
}

} // namespace mlx::core::gpu

