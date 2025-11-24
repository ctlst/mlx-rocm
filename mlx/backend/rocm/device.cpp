// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/rocm_utils.h"

#include <cstdlib>
#include <mutex>

namespace mlx::core::rocm {

namespace {

int get_max_ops_per_graph() {
  if (auto* env = std::getenv("MLX_MAX_OPS_PER_GRAPH")) {
    return std::atoi(env);
  }
  return 20;
}

int get_max_mb_per_graph() {
  if (auto* env = std::getenv("MLX_MAX_MB_PER_GRAPH")) {
    return std::atoi(env);
  }
  return 32;
}

} // namespace

CommandEncoder::CaptureContext::CaptureContext(CommandEncoder& enc)
    : graph(enc.device_), enc(enc) {
  CHECK_HIP_ERROR(hipStreamBeginCapture(enc.stream_, hipStreamCaptureModeGlobal));
}

CommandEncoder::CaptureContext::~CaptureContext() {
  if (!discard) {
    graph.end_capture(enc.stream_);
  }
}

CommandEncoder::ConcurrentContext::ConcurrentContext(CommandEncoder& enc)
    : enc(enc) {
  enc.in_concurrent_ = true;
  enc.concurrent_nodes_.clear();
}

CommandEncoder::ConcurrentContext::~ConcurrentContext() {
  enc.in_concurrent_ = false;
  if (!enc.concurrent_nodes_.empty()) {
    enc.insert_graph_dependencies(std::move(enc.concurrent_nodes_));
    enc.concurrent_nodes_.clear();
  }
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d),
      stream_(d),
      graph_(d),
      max_ops_per_graph_(get_max_ops_per_graph()),
      max_mb_per_graph_(get_max_mb_per_graph()) {}

void CommandEncoder::set_input_array(const array& arr) {
  if (arr.data_shared_ptr()) {
    active_deps_.push_back(reinterpret_cast<std::uintptr_t>(arr.data_shared_ptr().get()));
  }
}

void CommandEncoder::set_output_array(const array& arr) {
  if (arr.data_shared_ptr()) {
    active_outputs_.push_back(reinterpret_cast<std::uintptr_t>(arr.data_shared_ptr().get()));
    bytes_in_graph_ += arr.nbytes();
  }
}

void CommandEncoder::add_kernel_node(
    void* func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params) {
  // For simple execution without graphs, just launch the kernel
  hipLaunchKernelGGL(
      reinterpret_cast<void(*)(void)>(func),
      grid_dim,
      block_dim,
      smem_bytes,
      stream_,
      params);
}

void CommandEncoder::add_kernel_node(
    hipFunction_t func,
    dim3 grid_dim,
    dim3 block_dim,
    uint32_t smem_bytes,
    void** params) {
  CHECK_HIP_ERROR(hipModuleLaunchKernel(
      func,
      grid_dim.x, grid_dim.y, grid_dim.z,
      block_dim.x, block_dim.y, block_dim.z,
      smem_bytes,
      stream_,
      params,
      nullptr));
}

void CommandEncoder::add_graph_node(hipGraph_t child) {
  // For now, we execute the child graph immediately
  HipGraphExec exec;
  exec.instantiate(child);
  CHECK_HIP_ERROR(hipGraphLaunch(exec, stream_));
}

void CommandEncoder::insert_graph_dependencies(GraphNode node) {
  // Track node for dependency management
  for (auto dep : active_deps_) {
    auto it = node_map_.find(dep);
    if (it != node_map_.end()) {
      from_nodes_.push_back(it->second.node);
    }
  }
  for (auto out : active_outputs_) {
    node_map_[out] = node;
  }
  active_deps_.clear();
  active_outputs_.clear();
  node_count_++;
}

void CommandEncoder::insert_graph_dependencies(std::vector<GraphNode> nodes) {
  for (auto& node : nodes) {
    insert_graph_dependencies(std::move(node));
  }
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_.enqueue([this, task = std::move(task)]() {
    CHECK_HIP_ERROR(hipStreamSynchronize(stream_));
    task();
  });
}

bool CommandEncoder::needs_commit() {
  return node_count_ >= max_ops_per_graph_ ||
         bytes_in_graph_ >= static_cast<size_t>(max_mb_per_graph_) * 1024 * 1024;
}

void CommandEncoder::commit() {
  // Synchronize the stream
  CHECK_HIP_ERROR(hipStreamSynchronize(stream_));
  
  // Clear temporaries
  temporaries_.clear();
  
  // Reset state
  node_count_ = 0;
  bytes_in_graph_ = 0;
  node_map_.clear();
  from_nodes_.clear();
  to_nodes_.clear();
  graph_nodes_key_.clear();
  graph_deps_key_.clear();
  is_graph_updatable_ = true;
}

void CommandEncoder::synchronize() {
  commit();
  worker_.wait();
}

Device::Device(int device) : device_(device) {
  CHECK_HIP_ERROR(hipSetDevice(device_));
  
  // Get device properties
  hipDeviceProp_t props;
  CHECK_HIP_ERROR(hipGetDeviceProperties(&props, device_));
  device_name_ = props.name;
  
  // Parse GCN architecture version
  // For gfx1030, major=10, minor=3
  std::string arch = props.gcnArchName;
  if (arch.substr(0, 3) == "gfx") {
    int version = std::stoi(arch.substr(3));
    compute_capability_major_ = version / 100;
    compute_capability_minor_ = (version % 100) / 10;
  } else {
    compute_capability_major_ = 0;
    compute_capability_minor_ = 0;
  }
  
  // Initialize rocBLAS
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblas_));
}

Device::~Device() {
  if (rocblas_) {
    rocblas_destroy_handle(rocblas_);
  }
}

void Device::make_current() {
  CHECK_HIP_ERROR(hipSetDevice(device_));
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    it = encoders_.emplace(std::piecewise_construct,
                           std::forward_as_tuple(s.index),
                           std::forward_as_tuple(*this)).first;
  }
  return it->second;
}

namespace {
std::mutex device_mutex;
std::unordered_map<int, Device> devices;
} // namespace

Device& device(mlx::core::Device d) {
  std::lock_guard<std::mutex> lock(device_mutex);
  auto it = devices.find(d.index);
  if (it == devices.end()) {
    it = devices.emplace(std::piecewise_construct,
                         std::forward_as_tuple(d.index),
                         std::forward_as_tuple(d.index)).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  return device(s.device).get_command_encoder(s);
}

} // namespace mlx::core::rocm

