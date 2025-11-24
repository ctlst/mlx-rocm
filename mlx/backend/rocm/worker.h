// Copyright © 2025 Apple Inc.

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace mlx::core::rocm {

class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  void enqueue(std::function<void()> task);
  void wait();

 private:
  void run();

  std::thread thread_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  bool stop_{false};
  int pending_{0};
};

} // namespace mlx::core::rocm

