// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/primitives.h"

// roctracer is optional - only used for profiling markers
#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#else
#define roctxRangePush(x) ((void)0)
#define roctxRangePop() ((void)0)
#endif

namespace mlx::core {

// Gather and GatherAxis are in primitives.cpp as NO_GPU
// Scatter and ScatterAxis are in primitives.cpp as NO_GPU

// This file is for indexing-related operations that are implemented

} // namespace mlx::core

