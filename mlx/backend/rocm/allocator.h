// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/rocm/rocm_utils.h"

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <mutex>
#include <set>
#include <utility>

namespace mlx::core::rocm {

class CommandEncoder;

using allocator::Buffer;

// Stores HIP-managed unified memory.
struct HipBuffer {
  void* data;
  size_t size;
  int device; // -1 for managed
};

class SmallSizePool {
 private:
  union Block {
    Block* next;
    HipBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  Block* next_free_{nullptr};
  bool initialized_{false};
  
  void ensure_initialized();

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  HipBuffer* malloc();
  void free(HipBuffer* buf);
  bool in_pool(HipBuffer* buf);
};

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_async(size_t size, int device, hipStream_t stream);
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  void hip_free(HipBuffer* buf);
  void ensure_initialized();

  RocmAllocator();
  friend RocmAllocator& allocator();

  mutable std::mutex mutex_;
  size_t memory_limit_;
  size_t max_pool_size_;
  bool initialized_{false};
  BufferCache<HipBuffer> buffer_cache_{
      64 * 1024,  // page_size: 64KB
      [](HipBuffer* buf) { return buf->size; },
      [](HipBuffer* buf) {
        if (buf->data) {
          if (buf->device == -2) {
            std::free(buf->data);  // CPU allocation
          } else {
            hipFree(buf->data);  // HIP allocation
          }
        }
        delete buf;
      }};
  size_t active_memory_{0};
  size_t peak_memory_{0};
  std::vector<hipStream_t> free_streams_;
  SmallSizePool scalar_pool_;
};

RocmAllocator& allocator();

Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::rocm

