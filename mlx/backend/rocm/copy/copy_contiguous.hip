// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.cuh"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/common/copy.h"
#include "mlx/primitives.h"

namespace mlx::core::rocm {

// Kernel builders for copy operations
KernelBuilderResult build_copy_contiguous_kernel(Dtype dtype, bool use_int64) {
  std::string source = R"(
#include <hip/hip_runtime.h>

extern "C" {

)";

  // Generate kernel for the specific type
  std::string kernel_name;
  std::string type_name;
  std::string idx_type = use_int64 ? "int64_t" : "uint32_t";

  switch (dtype) {
    case bool_:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_bool_int64" : "copy_contiguous_v_kernel_bool_uint32";
      type_name = "bool";
      break;
    case int8:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_int8_int64" : "copy_contiguous_v_kernel_int8_uint32";
      type_name = "int8_t";
      break;
    case int16:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_int16_int64" : "copy_contiguous_v_kernel_int16_uint32";
      type_name = "int16_t";
      break;
    case int32:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_int32_int64" : "copy_contiguous_v_kernel_int32_uint32";
      type_name = "int32_t";
      break;
    case int64:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_int64_int64" : "copy_contiguous_v_kernel_int64_uint32";
      type_name = "int64_t";
      break;
    case uint8:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_uint8_int64" : "copy_contiguous_v_kernel_uint8_uint32";
      type_name = "uint8_t";
      break;
    case uint16:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_uint16_int64" : "copy_contiguous_v_kernel_uint16_uint32";
      type_name = "uint16_t";
      break;
    case uint32:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_uint32_int64" : "copy_contiguous_v_kernel_uint32_uint32";
      type_name = "uint32_t";
      break;
    case uint64:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_uint64_int64" : "copy_contiguous_v_kernel_uint64_uint32";
      type_name = "uint64_t";
      break;
    case float16:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_float16_int64" : "copy_contiguous_v_kernel_float16_uint32";
      type_name = "__half";
      break;
    case bfloat16:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_bfloat16_int64" : "copy_contiguous_v_kernel_bfloat16_uint32";
      type_name = "hip_bfloat16";
      break;
    case float32:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_float32_int64" : "copy_contiguous_v_kernel_float32_uint32";
      type_name = "float";
      break;
    case float64:
      kernel_name = use_int64 ? "copy_contiguous_v_kernel_float64_int64" : "copy_contiguous_v_kernel_float64_uint32";
      type_name = "double";
      break;
    default:
      throw std::runtime_error("Unsupported dtype for copy_contiguous kernel");
  }

  source += "__global__ void " + kernel_name + "(const " + type_name + "* __restrict__ src, " +
           type_name + "* __restrict__ dst, " + idx_type + " size) {\n";
  source += "  " + idx_type + " idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  source += "  if (idx < size) {\n";
  source += "    dst[idx] = src[idx];\n";
  source += "  }\n";
  source += "}\n\n";

  source += "}\n"; // extern "C"

  return {false, source, {kernel_name}};
}

void copy_contiguous_impl(
    CommandEncoder& encoder,
    const array& src,
    array& dst) {
  auto size = src.data_size();
  if (size == 0) {
    return;
  }

  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;
  bool use_int64 = size > UINT32_MAX;

  // Get or create JIT module for this kernel
  auto module_name = std::string("copy_contiguous_") + dtype_to_string(src.dtype()) +
                     (use_int64 ? "_int64" : "_uint32");
  auto& jit_module = get_jit_module(
      dst.device(), module_name,
      [dtype = src.dtype(), use_int64]() {
        return build_copy_contiguous_kernel(dtype, use_int64);
      });

  // Get the kernel function
  auto kernel_func = jit_module.get_kernel(module_name);

  // Prepare arguments
  KernelArgs args;
  args.append(src);  // source array
  args.append(dst);  // destination array
  if (use_int64) {
    args.append(static_cast<int64_t>(size));  // size as int64
  } else {
    args.append(static_cast<uint32_t>(size));  // size as uint32
  }

  // Launch the kernel
  encoder.add_kernel_node(
      kernel_func,
      dim3(num_blocks), dim3(block_size), 0, args.args());
}

void copy_contiguous(
    CommandEncoder& encoder,
    const array& src,
    array& dst) {
  // Use ROCm-specific dispatch that excludes complex64
  dispatch_all_types_rocm(src.dtype(), [&](auto type_tag) {
    copy_contiguous_impl(encoder, src, dst);
  });
}

} // namespace mlx::core::rocm

