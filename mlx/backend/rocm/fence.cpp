// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/scheduler.h"
#include "mlx/backend/rocm/device.h"

#include <hip/hip_runtime.h>
#include <condition_variable>
#include <mutex>

namespace mlx::core {

struct FenceImpl {
  uint32_t count{0};
  uint32_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
};

Fence::Fence(Stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  uint32_t count = f.count;
  
  if (stream.device == Device::gpu) {
    // For GPU, we need to synchronize through the scheduler
    scheduler::enqueue(stream, [count, fence_ = fence_]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f.mtx);
      if (f.value >= count) {
        return;
      }
      f.cv.wait(lk, [&f, count] { return f.value >= count; });
    });
  } else if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [count, fence_ = fence_]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f.mtx);
      if (f.value >= count) {
        return;
      }
      f.cv.wait(lk, [&f, count] { return f.value >= count; });
    });
  } else {
    throw std::runtime_error("[Fence::wait] Invalid stream.");
  }
}

void Fence::update(Stream stream, const array&, bool) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;
  uint32_t count = f.count;
  
  if (stream.device == Device::gpu) {
    // For GPU, signal through scheduler after GPU work completes
    scheduler::enqueue(stream, [count, fence_ = fence_]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f.mtx);
      f.value = count;
      f.cv.notify_all();
    });
  } else if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [count, fence_ = fence_]() mutable {
      auto& f = *static_cast<FenceImpl*>(fence_.get());
      std::unique_lock<std::mutex> lk(f.mtx);
      f.value = count;
      f.cv.notify_all();
    });
  } else {
    throw std::runtime_error("[Fence::update] Invalid stream.");
  }
}

} // namespace mlx::core

