// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/lru_cache.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <unordered_map>

namespace mlx::core::rocm {

class CommandEncoder {
 public:
  struct CaptureContext {
    CaptureContext(CommandEncoder& enc);
    ~CaptureContext();
    HipGraph graph;
    CommandEncoder& enc;
    bool discard{false};
  };
  struct ConcurrentContext {
    ConcurrentContext(CommandEncoder& enc);
    ~ConcurrentContext();
    CommandEncoder& enc;
  };

  explicit CommandEncoder(Device& d);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  CaptureContext capture_context() {
    return CaptureContext{*this};
  }
  ConcurrentContext concurrent_context() {
    return ConcurrentContext{*this};
  }

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  template <typename F, typename... Params>
  void add_kernel_node(
      F* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      Params&&... params) {
    constexpr size_t num = sizeof...(Params);
    void* ptrs[num];
    size_t i = 0;
    ([&](auto&& p) { ptrs[i++] = static_cast<void*>(&p); }(
         std::forward<Params>(params)),
     ...);
    add_kernel_node((void*)func, grid_dim, block_dim, smem_bytes, ptrs);
  }

  void add_kernel_node(
      hipFunction_t func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params);

  void add_kernel_node(
      void* func,
      dim3 grid_dim,
      dim3 block_dim,
      uint32_t smem_bytes,
      void** params);

  void add_graph_node(hipGraph_t child);

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  void add_completed_handler(std::function<void()> task);
  bool needs_commit();
  void commit();

  Device& device() {
    return device_;
  }

  hipStream_t stream() {
    return stream_;
  }

  // Wait until kernels and completion handlers are finished
  void synchronize();

 private:
  void add_kernel_node(const hipKernelNodeParams& params);

  struct GraphNode {
    hipGraphNode_t node;
    // K = kernel
    // E = empty
    // G* = subgraph (with metadata)
    // Symbols ':', '-' are reserved as separators
    std::string node_type;
    std::string id;
  };

  void insert_graph_dependencies(GraphNode node);
  void insert_graph_dependencies(std::vector<GraphNode> nodes);

  Device& device_;
  HipStream stream_;
  HipGraph graph_;
  Worker worker_;
  char node_count_{0};
  bool in_concurrent_{false};
  std::vector<hipGraphNode_t> from_nodes_;
  std::vector<hipGraphNode_t> to_nodes_;
  std::string graph_nodes_key_;
  std::string graph_deps_key_;
  std::vector<GraphNode> concurrent_nodes_;
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  LRUCache<std::string, HipGraphExec> graph_cache_;
  std::vector<std::uintptr_t> active_deps_;
  std::vector<std::uintptr_t> active_outputs_;
  std::unordered_map<std::uintptr_t, GraphNode> node_map_;
  size_t bytes_in_graph_{0};
  bool is_graph_updatable_{true};
  int max_ops_per_graph_;
  int max_mb_per_graph_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current HIP device, this method is thread-safe.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);

  int hip_device() const {
    return device_;
  }
  int compute_capability_major() const {
    return compute_capability_major_;
  }
  int compute_capability_minor() const {
    return compute_capability_minor_;
  }
  ::rocblas_handle get_rocblas_handle() const {
    return rocblas_;
  }

 private:
  int device_;
  int compute_capability_major_;
  int compute_capability_minor_;
  std::string device_name_;
  rocblas_handle rocblas_;
  std::unordered_map<int, CommandEncoder> encoders_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);

} // namespace mlx::core::rocm

