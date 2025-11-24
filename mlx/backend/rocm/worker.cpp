// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/worker.h"

namespace mlx::core::rocm {

Worker::Worker() : thread_([this]() { run(); }) {}

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_one();
  thread_.join();
}

void Worker::enqueue(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(std::move(task));
    pending_++;
  }
  cv_.notify_one();
}

void Worker::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  done_cv_.wait(lock, [this]() { return pending_ == 0; });
}

void Worker::run() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      pending_--;
    }
    done_cv_.notify_all();
  }
}

} // namespace mlx::core::rocm

