// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/jit_module.h"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
// roctracer is optional - only used for profiling markers
#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#else
#define roctxRangePush(x) ((void)0)
#define roctxRangePop() ((void)0)
#endif

namespace mlx::core {

namespace rocm {

// Kernel builder for arange operation
KernelBuilderResult build_arange_kernel(Dtype dtype) {
  std::string source = R"(
#include <hip/hip_runtime.h>

extern "C" {

)";

  // Generate kernel for the specific type
  std::string kernel_name;
  std::string type_name;

  switch (dtype) {
    case bool_:
      kernel_name = "arange_kernel_bool";
      type_name = "bool";
      break;
    case int8:
      kernel_name = "arange_kernel_int8";
      type_name = "int8_t";
      break;
    case int16:
      kernel_name = "arange_kernel_int16";
      type_name = "int16_t";
      break;
    case int32:
      kernel_name = "arange_kernel_int32";
      type_name = "int32_t";
      break;
    case int64:
      kernel_name = "arange_kernel_int64";
      type_name = "int64_t";
      break;
    case uint8:
      kernel_name = "arange_kernel_uint8";
      type_name = "uint8_t";
      break;
    case uint16:
      kernel_name = "arange_kernel_uint16";
      type_name = "uint16_t";
      break;
    case uint32:
      kernel_name = "arange_kernel_uint32";
      type_name = "uint32_t";
      break;
    case uint64:
      kernel_name = "arange_kernel_uint64";
      type_name = "uint64_t";
      break;
    case float16:
      kernel_name = "arange_kernel_float16";
      type_name = "__half";
      break;
    case bfloat16:
      kernel_name = "arange_kernel_bfloat16";
      type_name = "hip_bfloat16";
      break;
    case float32:
      kernel_name = "arange_kernel_float32";
      type_name = "float";
      break;
    case float64:
      kernel_name = "arange_kernel_float64";
      type_name = "double";
      break;
    default:
      throw std::runtime_error("Unsupported dtype for arange kernel");
  }

  source += "__global__ void " + kernel_name + "(" + type_name + "* out, " +
           type_name + " start, " + type_name + " step, size_t size) {\n";
  source += "  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
  source += "  if (idx < size) {\n";
  source += "    out[idx] = start + (" + type_name + ")idx * step;\n";
  source += "  }\n";
  source += "}\n\n";

  source += "}\n"; // extern "C"

  return {false, source, {kernel_name}};
}

} // namespace rocm

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  roctxRangePush("Arange::eval_gpu");
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  out.set_data(rocm::malloc_async(out.nbytes(), encoder));

  size_t size = out.size();
  if (size == 0) {
    roctxRangePop();
    return;
  }

  int block_size = 256;
  int num_blocks = (size + block_size - 1) / block_size;

  // Get or create JIT module for this kernel
  auto module_name = std::string("arange_") + dtype_to_string(out.dtype());
  auto& jit_module = rocm::get_jit_module(
      device(), module_name, [dtype = out.dtype()]() {
        return rocm::build_arange_kernel(dtype);
      });

  // Get the kernel function
  auto kernel_func = jit_module.get_kernel(module_name);

  // Prepare arguments
  rocm::KernelArgs args;
  args.append(out);  // output array
  args.append(static_cast<float>(start_));  // start value
  args.append(static_cast<float>(step_));  // step value
  args.append(size);  // size

  // Launch the kernel
  encoder.add_kernel_node(
      kernel_func,
      dim3(num_blocks), dim3(block_size), 0, args.args());

  roctxRangePop();
}

} // namespace mlx::core

