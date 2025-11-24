// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

bool is_available() {
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  return err == hipSuccess && device_count > 0;
}

} // namespace mlx::core::rocm

