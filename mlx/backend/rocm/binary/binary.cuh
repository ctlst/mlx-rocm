// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/common/binary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/binary_ops.cuh"
#include "mlx/backend/rocm/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
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

// Vector binary kernel
template <typename Op, typename In1, typename In2, typename Out, typename IdxT, int N_READS>
__global__ void binary_v(
    const In1* a,
    const In2* b,
    Out* out,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT vec_idx = index * N_READS;

  if ((index + 1) * N_READS <= size) {
    auto a_vec = load_vector<N_READS>(a, index);
    auto b_vec = load_vector<N_READS>(b, index);
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(Op{}(a_vec[i], b_vec[i]));
    }
    store_vector<N_READS>(out, index, out_vec);
  } else if (vec_idx < size) {
    for (IdxT i = vec_idx; i < size; ++i) {
      out[i] = static_cast<Out>(Op{}(a[i], b[i]));
    }
  }
}

// Scalar-vector binary kernel
template <typename Op, typename In1, typename In2, typename Out, typename IdxT, int N_READS>
__global__ void binary_sv(
    const In1* a,
    const In2* b,
    Out* out,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT vec_idx = index * N_READS;
  In1 a_val = *a;

  if ((index + 1) * N_READS <= size) {
    auto b_vec = load_vector<N_READS>(b, index);
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(Op{}(a_val, b_vec[i]));
    }
    store_vector<N_READS>(out, index, out_vec);
  } else if (vec_idx < size) {
    for (IdxT i = vec_idx; i < size; ++i) {
      out[i] = static_cast<Out>(Op{}(a_val, b[i]));
    }
  }
}

// Vector-scalar binary kernel
template <typename Op, typename In1, typename In2, typename Out, typename IdxT, int N_READS>
__global__ void binary_vs(
    const In1* a,
    const In2* b,
    Out* out,
    IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  IdxT vec_idx = index * N_READS;
  In2 b_val = *b;

  if ((index + 1) * N_READS <= size) {
    auto a_vec = load_vector<N_READS>(a, index);
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = static_cast<Out>(Op{}(a_vec[i], b_val));
    }
    store_vector<N_READS>(out, index, out_vec);
  } else if (vec_idx < size) {
    for (IdxT i = vec_idx; i < size; ++i) {
      out[i] = static_cast<Out>(Op{}(a[i], b_val));
    }
  }
}

// General binary kernel
template <typename Op, typename In1, typename In2, typename Out, typename IdxT>
__global__ void binary_g(
    const In1* a,
    const In2* b,
    Out* out,
    IdxT size,
    const int32_t* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    IdxT a_idx = elem_to_loc(index, shape, a_strides, ndim);
    IdxT b_idx = elem_to_loc(index, shape, b_strides, ndim);
    out[index] = static_cast<Out>(Op{}(a[a_idx], b[b_idx]));
  }
}

template <typename Op, typename In, typename Out>
constexpr bool supports_binary_op() {
  if (std::is_same_v<Op, Add> || std::is_same_v<Op, Divide> ||
      std::is_same_v<Op, Maximum> || std::is_same_v<Op, Minimum> ||
      std::is_same_v<Op, Multiply> || std::is_same_v<Op, Subtract> ||
      std::is_same_v<Op, Power> || std::is_same_v<Op, Remainder>) {
    return std::is_same_v<In, Out>;
  }
  if (std::is_same_v<Op, Equal> || std::is_same_v<Op, Greater> ||
      std::is_same_v<Op, GreaterEqual> || std::is_same_v<Op, Less> ||
      std::is_same_v<Op, LessEqual> || std::is_same_v<Op, NotEqual>) {
    return std::is_same_v<Out, bool>;
  }
  if (std::is_same_v<Op, LogicalAnd> || std::is_same_v<Op, LogicalOr>) {
    return std::is_same_v<Out, bool> && std::is_same_v<In, bool>;
  }
  if (std::is_same_v<Op, NaNEqual>) {
    return std::is_same_v<Out, bool> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, LogAddExp>) {
    return std::is_same_v<In, Out> && is_inexact_v<In>;
  }
  if (std::is_same_v<Op, ArcTan2>) {
    return std::is_same_v<In, Out> && is_floating_v<In>;
  }
  if (std::is_same_v<Op, BitwiseAnd> || std::is_same_v<Op, BitwiseOr> ||
      std::is_same_v<Op, BitwiseXor>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In>;
  }
  if (std::is_same_v<Op, LeftShift> || std::is_same_v<Op, RightShift>) {
    return std::is_same_v<In, Out> && std::is_integral_v<In> &&
        !std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace rocm

template <typename Op>
void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  
  if (out.size() == 0) {
    return;
  }
  
  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  auto bopt = get_binary_op_type(a, b);
  bool large = out.size() > UINT32_MAX;
  
  // Use ROCm-specific dispatch that excludes complex64
  rocm::dispatch_all_types_rocm(a.dtype(), [&](auto in_type_tag) {
    rocm::dispatch_all_types_rocm(out.dtype(), [&](auto out_type_tag) {
      using In_T = typename decltype(in_type_tag)::type;
      using Out_T = typename decltype(out_type_tag)::type;

      if constexpr (rocm::supports_binary_op<Op, In_T, Out_T>()) {

        using InType = rocm::hip_type_t<In_T>;
        using OutType = rocm::hip_type_t<Out_T>;
        
        constexpr int N_READS = 4;
        int block_size = 256;
        size_t size = out.size();
        int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
        
        switch (bopt) {
          case BinaryOpType::ScalarScalar:
          case BinaryOpType::VectorVector:
            if (large) {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              int64_t size_val = static_cast<int64_t>(size);
              rocm::rocm_launch_kernel(
                  rocm::binary_v<Op, InType, InType, OutType, int64_t, N_READS>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val);
            } else {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              uint32_t size_val = static_cast<uint32_t>(size);
              rocm::rocm_launch_kernel(
                  rocm::binary_v<Op, InType, InType, OutType, uint32_t, N_READS>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val);
            }
            break;
          case BinaryOpType::ScalarVector:
            if (large) {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              int64_t size_val = static_cast<int64_t>(size);
              rocm::rocm_launch_kernel(
                  rocm::binary_sv<Op, InType, InType, OutType, int64_t, N_READS>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val);
            } else {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              uint32_t size_val = static_cast<uint32_t>(size);
              rocm::rocm_launch_kernel(
                  rocm::binary_sv<Op, InType, InType, OutType, uint32_t, N_READS>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val);
            }
            break;
          case BinaryOpType::VectorScalar:
            if (large) {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              int64_t size_val = static_cast<int64_t>(size);
              rocm::rocm_launch_kernel(
                  rocm::binary_vs<Op, InType, InType, OutType, int64_t, N_READS>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val);
            } else {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              uint32_t size_val = static_cast<uint32_t>(size);
              rocm::rocm_launch_kernel(
                  rocm::binary_vs<Op, InType, InType, OutType, uint32_t, N_READS>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val);
            }
            break;
          case BinaryOpType::General: {
            num_blocks = (size + block_size - 1) / block_size;
            rocm::ConstParam<int32_t> shape_param(out.shape());
            rocm::ConstParam<int64_t> a_strides_param;
            rocm::ConstParam<int64_t> b_strides_param;
            for (int i = 0; i < out.ndim() && i < rocm::MAX_NDIM; ++i) {
              a_strides_param.data[i] = a.strides()[i];
              b_strides_param.data[i] = b.strides()[i];
            }
            if (large) {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              int64_t size_val = static_cast<int64_t>(size);
              int32_t* shape_data = shape_param.data;
              int64_t* a_strides_data = a_strides_param.data;
              int64_t* b_strides_data = b_strides_param.data;
              int ndim_val = out.ndim();
              rocm::rocm_launch_kernel(
                  rocm::binary_g<Op, InType, InType, OutType, int64_t>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val, shape_data, a_strides_data, b_strides_data, ndim_val);
            } else {
              auto* a_ptr = rocm::gpu_ptr<InType>(a);
              auto* b_ptr = rocm::gpu_ptr<InType>(b);
              auto* out_ptr = rocm::gpu_ptr<OutType>(out);
              uint32_t size_val = static_cast<uint32_t>(size);
              int32_t* shape_data = shape_param.data;
              int64_t* a_strides_data = a_strides_param.data;
              int64_t* b_strides_data = b_strides_param.data;
              int ndim_val = out.ndim();
              rocm::rocm_launch_kernel(
                  rocm::binary_g<Op, InType, InType, OutType, uint32_t>,
                  dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                  a_ptr, b_ptr, out_ptr, size_val, shape_data, a_strides_data, b_strides_data, ndim_val);
            }
            break;
          }
        }
      } // close if constexpr
    }); // close dispatch_all_types_rocm(out.dtype(), ...)
  }); // close dispatch_all_types_rocm(a.dtype(), ...)
}

template <typename Op>
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);
  auto& encoder = rocm::get_command_encoder(s);
  set_binary_op_output_data(
      a, b, out, bopt, [&](auto n) { return rocm::malloc_async(n, encoder); });
  binary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define BINARY_GPU(func)                                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    roctxRangePush(#func "::eval_gpu");                               \
    auto& s = out.primitive().stream();                               \
    binary_op_gpu<rocm::func>(inputs, out, name(), s);                \
    roctxRangePop();                                                  \
  }

} // namespace mlx::core

