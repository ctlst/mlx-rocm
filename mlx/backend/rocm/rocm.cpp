// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm.h"

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <string>

namespace mlx::core::rocm {

bool is_available() {
  // Use cached result from gpu::is_available() or check directly
  static int result = -1;  // -1 = not checked, 0 = not available, 1 = available
  if (result >= 0) {
    return result == 1;
  }
  
  // Check environment variable to force disable GPU
  if (const char* env = std::getenv("MLX_DISABLE_GPU")) {
    if (std::string(env) == "1" || std::string(env) == "true") {
      result = 0;
      return false;
    }
  }
  
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  
  if (err == hipSuccess && device_count > 0) {
    result = 1;
    return true;
  }
  
  result = 0;
  return false;
}

} // namespace mlx::core::rocm

