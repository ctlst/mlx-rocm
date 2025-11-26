// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>

namespace mlx::core::rocm {

namespace {

// Scalar buffer pool settings
constexpr size_t kPoolSize = 4096;
constexpr size_t kScalarSize = 16;

// For now, use CPU-only allocation in the common allocator path
// GPU memory will be allocated separately when kernels actually run
// This avoids HIP initialization issues during array creation

// Get system memory info - use a safe default, no HIP calls
size_t get_system_memory() {
  // Return 8GB as default - no HIP calls to avoid segfaults
  return 8ULL * 1024 * 1024 * 1024;
}

} // namespace

// SmallSizePool is disabled - not using HIP memory pool
SmallSizePool::SmallSizePool() 
    : buffer_(nullptr), data_(nullptr), next_free_(nullptr), initialized_(false) {
}

void SmallSizePool::ensure_initialized() {
  // Disabled - using regular malloc instead
  initialized_ = true;
}

SmallSizePool::~SmallSizePool() {
}

HipBuffer* SmallSizePool::malloc() {
  // Disabled - always return nullptr to use regular malloc path
  return nullptr;
}

void SmallSizePool::free(HipBuffer* buf) {
  // Not used
}

bool SmallSizePool::in_pool(HipBuffer* buf) {
  return false;
}

RocmAllocator::RocmAllocator()
    : memory_limit_(0),
      max_pool_size_(0),
      initialized_(false) {}

void RocmAllocator::ensure_initialized() {
  if (initialized_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return;  // Double-check after acquiring lock
  }
  memory_limit_ = get_system_memory();
  max_pool_size_ = memory_limit_ / 2;
  initialized_ = true;
}

Buffer RocmAllocator::malloc(size_t size) {
  ensure_initialized();
  
  // Use CPU allocation for now - GPU memory is allocated separately
  // when kernels run via hipMalloc in the kernel launch path
  void* ptr = std::malloc(size);
  if (!ptr && size > 0) {
    throw std::runtime_error(
        "[ROCm] Failed to allocate " + std::to_string(size) + " bytes");
  }
  auto* buf = new HipBuffer{ptr, size, -2};  // -2 indicates CPU allocation
  
  std::lock_guard<std::mutex> lock(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(peak_memory_, active_memory_);
  return Buffer{buf};
}

Buffer RocmAllocator::malloc_async(size_t size, int device, hipStream_t stream) {
  // For async allocation, we use regular malloc for now
  // HIP async memory management is more limited than CUDA
  return malloc(size);
}

void RocmAllocator::free(Buffer buffer) {
  auto* buf = static_cast<HipBuffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Check if it's from the scalar pool
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
    return;
  }

  active_memory_ -= buf->size;

  // Add to cache for reuse
  if (buffer_cache_.cache_size() + buf->size <= max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    hip_free(buf);
  }
}

void RocmAllocator::hip_free(HipBuffer* buf) {
  if (buf->data) {
    // All allocations are now CPU-based (std::malloc)
    std::free(buf->data);
  }
  delete buf;
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<HipBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

size_t RocmAllocator::get_active_memory() const {
  return active_memory_;
}

size_t RocmAllocator::get_peak_memory() const {
  return peak_memory_;
}

void RocmAllocator::reset_peak_memory() {
  std::lock_guard<std::mutex> lock(mutex_);
  peak_memory_ = active_memory_;
}

size_t RocmAllocator::get_memory_limit() {
  if (!initialized_) {
    // Return a large default so memory checks pass before GPU init
    return SIZE_MAX;
  }
  return memory_limit_;
}

size_t RocmAllocator::set_memory_limit(size_t limit) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t old_limit = memory_limit_;
  memory_limit_ = limit;
  return old_limit;
}

size_t RocmAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t RocmAllocator::set_cache_limit(size_t limit) {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t old_limit = max_pool_size_;
  max_pool_size_ = limit;
  return old_limit;
}

void RocmAllocator::clear_cache() {
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_cache_.clear();
}

RocmAllocator& allocator() {
  // Use pointer to avoid destructor being called on exit
  // which can cause issues with HIP shutdown order
  static RocmAllocator* allocator_ = new RocmAllocator;
  return *allocator_;
}

Buffer malloc_async(size_t size, CommandEncoder& encoder) {
  return allocator().malloc_async(size, encoder.device().hip_device(), encoder.stream());
}

} // namespace mlx::core::rocm

namespace mlx::core::allocator {

Allocator& allocator() {
  return rocm::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  auto& buf = *static_cast<rocm::HipBuffer*>(ptr_);
  return buf.data;
}

} // namespace mlx::core::allocator

// Global memory functions required by mlx::core
namespace mlx::core {

size_t get_active_memory() {
  return rocm::allocator().get_active_memory();
}

size_t get_peak_memory() {
  return rocm::allocator().get_peak_memory();
}

void reset_peak_memory() {
  return rocm::allocator().reset_peak_memory();
}

size_t set_memory_limit(size_t limit) {
  return rocm::allocator().set_memory_limit(limit);
}

size_t get_memory_limit() {
  return rocm::allocator().get_memory_limit();
}

size_t get_cache_memory() {
  return rocm::allocator().get_cache_memory();
}

size_t set_cache_limit(size_t limit) {
  return rocm::allocator().set_cache_limit(limit);
}

void clear_cache() {
  rocm::allocator().clear_cache();
}

size_t set_wired_limit(size_t) {
  // Not applicable for ROCm
  return 0;
}

} // namespace mlx::core

