// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm_utils.h"

#include <stdexcept>
#include <string>

namespace mlx::core {

void check_rocblas_error(const char* name, rocblas_status err) {
  if (err != rocblas_status_success) {
    throw std::runtime_error(
        std::string("[ROCm] rocBLAS error in ") + name + ": " +
        rocblas_status_to_string(err));
  }
}

void check_hip_error(const char* name, hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("[ROCm] HIP error in ") + name + ": " +
        hipGetErrorString(err));
  }
}

HipGraph::HipGraph(rocm::Device& device) {
  CHECK_HIP_ERROR(hipGraphCreate(&handle_, 0));
}

void HipGraph::end_capture(hipStream_t stream) {
  CHECK_HIP_ERROR(hipStreamEndCapture(stream, &handle_));
}

void HipGraphExec::instantiate(hipGraph_t graph) {
  reset();
  CHECK_HIP_ERROR(hipGraphInstantiate(&handle_, graph, nullptr, nullptr, 0));
}

HipStream::HipStream(rocm::Device& device) {
  CHECK_HIP_ERROR(
      hipStreamCreateWithFlags(&handle_, hipStreamNonBlocking));
}

} // namespace mlx::core

