// Copyright © 2025 Apple Inc.

#pragma once

namespace mlx::core::rocm {

// Maximum number of dimensions supported.
constexpr int MAX_NDIM = 8;

// Default block size for HIP kernels.
// gfx1030 uses wave64 (64 threads per wavefront)
constexpr int DEFAULT_BLOCK_SIZE = 256;

// Maximum shared memory per block (in bytes)
constexpr int MAX_SHARED_MEM = 65536;

} // namespace mlx::core::rocm

