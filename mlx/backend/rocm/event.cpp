// Copyright © 2025 Apple Inc.

#include "mlx/event.h"
#include "mlx/scheduler.h"
#include "mlx/backend/rocm/device.h"

#include <hip/hip_runtime.h>
#include <condition_variable>
#include <mutex>

namespace mlx::core {

struct EventCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
  hipEvent_t hip_event{nullptr};
};

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) {
    auto* ec = static_cast<EventCounter*>(ptr);
    if (ec->hip_event) {
      hipEventDestroy(ec->hip_event);
    }
    delete ec;
  };
  auto* ec = new EventCounter{};
  hipEventCreate(&ec->hip_event);
  event_ = std::shared_ptr<void>(ec, dtor);
}

void Event::wait() {
  auto ec = static_cast<EventCounter*>(event_.get());
  
  // If there's a HIP event, wait on it
  if (ec->hip_event) {
    hipEventSynchronize(ec->hip_event);
  }
  
  // Also check the counter
  std::unique_lock<std::mutex> lk(ec->mtx);
  if (ec->value >= value()) {
    return;
  }
  ec->cv.wait(lk, [value = value(), ec] { return ec->value >= value; });
}

void Event::wait(Stream stream) {
  auto ec = static_cast<EventCounter*>(event_.get());
  
  if (stream.device == Device::gpu && ec->hip_event) {
    // Make GPU stream wait on the event
    auto& d = rocm::device(stream.device);
    hipStreamWaitEvent(d.get_stream(stream.index), ec->hip_event, 0);
  } else {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  }
}

void Event::signal(Stream stream) {
  auto ec = static_cast<EventCounter*>(event_.get());
  
  if (stream.device == Device::gpu && ec->hip_event) {
    // Record event on GPU stream
    auto& d = rocm::device(stream.device);
    hipEventRecord(ec->hip_event, d.get_stream(stream.index));
  }
  
  scheduler::enqueue(stream, [*this]() mutable {
    auto ec = static_cast<EventCounter*>(event_.get());
    {
      std::lock_guard<std::mutex> lk(ec->mtx);
      ec->value = value();
    }
    ec->cv.notify_all();
  });
}

bool Event::is_signaled() const {
  auto ec = static_cast<EventCounter*>(event_.get());
  
  // Check HIP event first
  if (ec->hip_event) {
    hipError_t status = hipEventQuery(ec->hip_event);
    if (status == hipSuccess) {
      return true;
    }
  }
  
  // Fall back to counter check
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    return (ec->value >= value());
  }
}

} // namespace mlx::core

