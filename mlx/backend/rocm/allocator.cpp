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

// Get GPU memory info safely
size_t get_gpu_memory() {
  size_t free, total;
  hipError_t err = hipMemGetInfo(&free, &total);
  if (err == hipSuccess) {
    return total;
  }
  // Fallback to 8GB if HIP call fails
  return 8ULL * 1024 * 1024 * 1024;
}

} // namespace

SmallSizePool::SmallSizePool()
    : buffer_(nullptr), data_(nullptr), next_free_(nullptr), initialized_(false) {
}

void SmallSizePool::ensure_initialized() {
  if (initialized_) {
    return;
  }

  auto num_blocks = kPoolSize / kScalarSize;
  buffer_ = new Block[num_blocks];
  next_free_ = buffer_;

  // Allocate GPU memory for the pool
  CHECK_HIP_ERROR(hipMalloc(&data_, kPoolSize));

  auto curr = next_free_;
  for (size_t i = 1; i < num_blocks; ++i) {
    curr->next = buffer_ + i;
    curr = curr->next;
  }
  curr->next = nullptr;

  initialized_ = true;
}

SmallSizePool::~SmallSizePool() {
  if (data_) {
    hipFree(data_);
  }
  delete[] buffer_;
}

HipBuffer* SmallSizePool::malloc() {
  ensure_initialized();

  if (next_free_ == nullptr) {
    return nullptr;
  }

  Block* b = next_free_;
  uint64_t i = next_free_ - buffer_;
  next_free_ = next_free_->next;

  b->buf.data = static_cast<char*>(data_) + i * kScalarSize;
  b->buf.size = kScalarSize;
  b->buf.device = -2;  // Mark as pool allocation
  return &b->buf;
}

void SmallSizePool::free(HipBuffer* buf) {
  auto b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(HipBuffer* buf) {
  auto b = reinterpret_cast<Block*>(buf);
  int64_t block_num = b - buffer_;
  auto num_blocks = kPoolSize / kScalarSize;
  return block_num >= 0 && block_num < static_cast<int64_t>(num_blocks);
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
  memory_limit_ = get_gpu_memory();
  max_pool_size_ = memory_limit_ * 0.9;  // Use 90% of available GPU memory
  initialized_ = true;
}

Buffer RocmAllocator::malloc(size_t size) {
  ensure_initialized();

  // Try small pool first for small allocations
  if (size <= kScalarSize) {
    auto* buf = scalar_pool_.malloc();
    if (buf) {
      std::lock_guard<std::mutex> lock(mutex_);
      active_memory_ += size;
      peak_memory_ = std::max(peak_memory_, active_memory_);
      return Buffer{buf};
    }
  }

  // Allocate GPU memory
  void* ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, size);
  if (err != hipSuccess || (!ptr && size > 0)) {
    throw std::runtime_error(
        "[ROCm] Failed to allocate " + std::to_string(size) + " bytes of GPU memory");
  }
  auto* buf = new HipBuffer{ptr, size, 0};  // 0 indicates GPU allocation

  std::lock_guard<std::mutex> lock(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(peak_memory_, active_memory_);
  return Buffer{buf};
}

Buffer RocmAllocator::malloc_async(size_t size, int device, hipStream_t stream) {
  ensure_initialized();

  // Allocate GPU memory asynchronously if possible
  void* ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, size);
  if (err != hipSuccess || (!ptr && size > 0)) {
    throw std::runtime_error(
        "[ROCm] Failed to allocate " + std::to_string(size) + " bytes of GPU memory asynchronously");
  }
  auto* buf = new HipBuffer{ptr, size, device};

  std::lock_guard<std::mutex> lock(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(peak_memory_, active_memory_);
  return Buffer{buf};
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
    if (buf->device == -2) {
      // This was allocated from the scalar pool, let the pool handle it
      // (but this shouldn't happen since pool buffers are handled separately)
    } else {
      // GPU allocation
      hipFree(buf->data);
    }
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

