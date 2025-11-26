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

// Global flag to track if HIP is working
static bool g_hip_checked = false;
static bool g_hip_available = false;

bool check_hip_available() {
  if (g_hip_checked) {
    return g_hip_available;
  }
  g_hip_checked = true;
  
  // Try basic HIP init
  hipError_t err = hipInit(0);
  if (err != hipSuccess) {
    g_hip_available = false;
    return false;
  }
  
  int device_count = 0;
  err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    g_hip_available = false;
    return false;
  }
  
  g_hip_available = true;
  return true;
}

// Get system memory info - called lazily
size_t get_system_memory() {
  if (!check_hip_available()) {
    return 8ULL * 1024 * 1024 * 1024;
  }
  
  // Set device 0 as current
  hipError_t set_err = hipSetDevice(0);
  if (set_err != hipSuccess) {
    return 8ULL * 1024 * 1024 * 1024;
  }
  
  // Now query memory
  size_t free_mem = 0, total_mem = 0;
  hipError_t err = hipMemGetInfo(&free_mem, &total_mem);
  if (err == hipSuccess && total_mem > 0) {
    return total_mem;
  }
  // Fallback to 8GB if we can't query
  return 8ULL * 1024 * 1024 * 1024;
}

} // namespace

SmallSizePool::SmallSizePool() 
    : buffer_(nullptr), data_(nullptr), next_free_(nullptr), initialized_(false) {
  // Lazy initialization - don't allocate HIP memory until first use
}

void SmallSizePool::ensure_initialized() {
  if (initialized_) {
    return;
  }
  
  // Allocate pool of small buffers
  size_t pool_bytes = kPoolSize * sizeof(Block);
  buffer_ = static_cast<Block*>(std::malloc(pool_bytes));
  
  hipError_t err = hipMallocManaged(&data_, kPoolSize * kScalarSize);
  if (err != hipSuccess) {
    std::free(buffer_);
    buffer_ = nullptr;
    data_ = nullptr;
    // Don't throw - just disable the pool
    initialized_ = true;
    return;
  }
  
  // Initialize free list
  for (size_t i = 0; i < kPoolSize; ++i) {
    buffer_[i].buf.data = static_cast<char*>(data_) + i * kScalarSize;
    buffer_[i].buf.size = kScalarSize;
    buffer_[i].buf.device = -1;
    buffer_[i].next = (i + 1 < kPoolSize) ? &buffer_[i + 1] : nullptr;
  }
  next_free_ = buffer_;
  initialized_ = true;
}

SmallSizePool::~SmallSizePool() {
  if (data_) {
    hipFree(data_);
    data_ = nullptr;
  }
  if (buffer_) {
    std::free(buffer_);
    buffer_ = nullptr;
  }
}

HipBuffer* SmallSizePool::malloc() {
  ensure_initialized();
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* block = next_free_;
  next_free_ = block->next;
  return &block->buf;
}

void SmallSizePool::free(HipBuffer* buf) {
  Block* block = reinterpret_cast<Block*>(buf);
  block->next = next_free_;
  next_free_ = block;
}

bool SmallSizePool::in_pool(HipBuffer* buf) {
  if (!initialized_ || buffer_ == nullptr) {
    return false;
  }
  return buf >= &buffer_[0].buf && 
         buf < &buffer_[kPoolSize].buf;
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
  
  // Try to initialize HIP - if it fails, we'll use fallback memory handling
  try {
    memory_limit_ = get_system_memory();
    max_pool_size_ = memory_limit_ / 2;
  } catch (...) {
    // HIP initialization failed, use defaults
    memory_limit_ = 8ULL * 1024 * 1024 * 1024;  // 8GB
    max_pool_size_ = memory_limit_ / 2;
  }
  initialized_ = true;
}

Buffer RocmAllocator::malloc(size_t size) {
  ensure_initialized();
  
  // If HIP is not available, fall back to CPU allocation
  if (!check_hip_available()) {
    void* ptr = std::malloc(size);
    if (!ptr && size > 0) {
      throw std::runtime_error(
          "[ROCm] Failed to allocate " + std::to_string(size) + " bytes (CPU fallback)");
    }
    auto* buf = new HipBuffer{ptr, size, -2};  // -2 indicates CPU allocation
    std::lock_guard<std::mutex> lock(mutex_);
    active_memory_ += size;
    peak_memory_ = std::max(peak_memory_, active_memory_);
    return Buffer{buf};
  }
  
  // For very small allocations, use the scalar pool
  if (size <= kScalarSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto* buf = scalar_pool_.malloc()) {
      return Buffer{buf};
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // Check the buffer cache first
  if (auto buf = buffer_cache_.reuse_from_cache(size)) {
    active_memory_ += buf->size;
    peak_memory_ = std::max(peak_memory_, active_memory_);
    return Buffer{buf};
  }

  // Check memory limit
  if (active_memory_ + size > memory_limit_) {
    // Try to clear cache to make room
    buffer_cache_.clear();
    if (active_memory_ + size > memory_limit_) {
      throw std::runtime_error(
          "[ROCm] Memory limit exceeded. Requested " + 
          std::to_string(size) + " bytes, but only " +
          std::to_string(memory_limit_ - active_memory_) + 
          " bytes available.");
    }
  }

  // Allocate new buffer
  void* ptr = nullptr;
  hipError_t err = hipMallocManaged(&ptr, size);
  if (err != hipSuccess) {
    // Try to clear cache and retry
    buffer_cache_.clear();
    err = hipMallocManaged(&ptr, size);
    if (err != hipSuccess) {
      throw std::runtime_error(
          "[ROCm] Failed to allocate " + std::to_string(size) +
          " bytes: " + hipGetErrorString(err));
    }
  }

  auto* buf = new HipBuffer{ptr, size, -1};
  active_memory_ += size;
  peak_memory_ = std::max(peak_memory_, active_memory_);
  
  return Buffer{buf};
}

Buffer RocmAllocator::malloc_async(size_t size, int device, hipStream_t stream) {
  ensure_initialized();
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
    if (buf->device == -2) {
      // CPU allocation - use std::free
      std::free(buf->data);
    } else {
      // HIP allocation
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

