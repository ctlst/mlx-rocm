// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/common/unary.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/device/unary_ops.cuh"
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

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void unary_v(const In* in, Out* out, IdxT size) {
  IdxT index = blockIdx.x * blockDim.x + threadIdx.x;

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = Op{}(in[i]);
    }
  } else {
    auto in_vec = load_vector<N_READS>(in, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(in_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename Op, typename In, typename Out, typename IdxT, int N_READS>
__global__ void unary_g(
    const In* in,
    Out* out,
    IdxT size_rest,
    const int32_t* shape,
    const int64_t* strides,
    int ndim) {
  IdxT index_rest = blockIdx.y * blockDim.y + threadIdx.y;
  if (index_rest >= size_rest) {
    return;
  }

  auto shape_x = shape[ndim - 1];
  auto stride_x = strides[ndim - 1];
  IdxT index_x = blockIdx.x * blockDim.x + threadIdx.x;
  auto idx = elem_to_loc(index_rest * shape_x, shape, strides, ndim);
  
  if (index_x * N_READS < shape_x) {
    auto in_vec = load_vector<N_READS>(
        in + idx, index_x, shape_x, stride_x, In(0));
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = Op{}(in_vec[i]);
    }
    store_vector(out + shape_x * index_rest, index_x, out_vec, shape_x);
  }
}

template <typename Op, typename In, typename Out>
constexpr bool supports_unary_op() {
  if constexpr (std::is_same_v<Op, Abs> || std::is_same_v<Op, Negative> ||
      std::is_same_v<Op, Sign> || std::is_same_v<Op, Square>) {
    return std::is_same_v<In, Out>;
  }
  if constexpr (std::is_same_v<Op, ArcCosh> || std::is_same_v<Op, ArcSinh> ||
      std::is_same_v<Op, ArcTanh> || std::is_same_v<Op, Erf> ||
      std::is_same_v<Op, ErfInv> || std::is_same_v<Op, Expm1> ||
      std::is_same_v<Op, Sigmoid>) {
    return std::is_same_v<In, Out> && std::is_floating_point_v<In>;
  }
  if constexpr (std::is_same_v<Op, Ceil> || std::is_same_v<Op, Floor>) {
    return std::is_same_v<In, Out>;
  }
  if constexpr (std::is_same_v<Op, ArcCos> || std::is_same_v<Op, ArcSin> ||
      std::is_same_v<Op, ArcTan> || std::is_same_v<Op, Cos> ||
      std::is_same_v<Op, Cosh> || std::is_same_v<Op, Exp> ||
      std::is_same_v<Op, Log> || std::is_same_v<Op, Log1p> ||
      std::is_same_v<Op, Round> || std::is_same_v<Op, Rsqrt> ||
      std::is_same_v<Op, Sqrt> || std::is_same_v<Op, Sin> ||
      std::is_same_v<Op, Sinh> || std::is_same_v<Op, Tan> ||
      std::is_same_v<Op, Tanh>) {
    return std::is_same_v<In, Out>;
  }
  if constexpr (std::is_same_v<Op, LogicalNot>) {
    return std::is_same_v<In, Out> && std::is_same_v<In, bool>;
  }
  return false;
}

} // namespace rocm

template <typename Op>
void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& in = inputs[0];
  if (in.size() == 0) {
    return;
  }
  bool contig = in.flags().contiguous;
  bool large;
  if (!contig) {
    large = in.data_size() > INT32_MAX || out.size() > INT32_MAX;
  } else {
    large = in.data_size() > UINT32_MAX;
  }

  auto& encoder = rocm::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Use ROCm-specific dispatch that excludes complex64
  rocm::dispatch_all_types_rocm(in.dtype(), [&](auto in_type_tag) {
    rocm::dispatch_all_types_rocm(out.dtype(), [&](auto out_type_tag) {
      using CTYPE_IN = typename decltype(in_type_tag)::type;
      using CTYPE_OUT = typename decltype(out_type_tag)::type;
      if constexpr (rocm::supports_unary_op<Op, CTYPE_IN, CTYPE_OUT>()) {
        using InType = rocm::hip_type_t<CTYPE_IN>;
        using OutType = rocm::hip_type_t<CTYPE_OUT>;
        
        if (contig) {
          constexpr int N_READS = 4;
          int block_size = 256;
          size_t size = out.data_size();
          int num_blocks = (size + block_size * N_READS - 1) / (block_size * N_READS);
          
          if (large) {
            hipLaunchKernelGGL(
                (rocm::unary_v<Op, InType, OutType, int64_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                rocm::gpu_ptr<InType>(in),
                rocm::gpu_ptr<OutType>(out),
                static_cast<int64_t>(size));
          } else {
            hipLaunchKernelGGL(
                (rocm::unary_v<Op, InType, OutType, uint32_t, N_READS>),
                dim3(num_blocks), dim3(block_size), 0, encoder.stream(),
                rocm::gpu_ptr<InType>(in),
                rocm::gpu_ptr<OutType>(out),
                static_cast<uint32_t>(size));
          }
        } else {
          auto [shape, strides] = rocm::collapse_contiguous_dims(in);
          auto ndim = shape.size();
          int work_per_thread = 1;
          auto dim0 = ndim > 0 ? shape.back() : 1;
          auto rest = out.size() / dim0;
          
          if (dim0 >= 4) {
            work_per_thread = 4;
          }
          dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
          
          auto block_dims = rocm::get_block_dims(dim0, rest, 1);
          uint32_t num_blocks_x = (dim0 + block_dims.x - 1) / block_dims.x;
          uint32_t num_blocks_y = (rest + block_dims.y - 1) / block_dims.y;
          
          rocm::ConstParam<int32_t> shape_param(shape);
          rocm::ConstParam<int64_t> strides_param;
          for (size_t i = 0; i < strides.size() && i < rocm::MAX_NDIM; ++i) {
            strides_param.data[i] = strides[i];
          }
          
          if (work_per_thread == 4) {
            if (large) {
              hipLaunchKernelGGL(
                  (rocm::unary_g<Op, InType, OutType, int64_t, 4>),
                  dim3(num_blocks_x, num_blocks_y), block_dims, 0, encoder.stream(),
                  rocm::gpu_ptr<InType>(in),
                  rocm::gpu_ptr<OutType>(out),
                  static_cast<int64_t>(rest),
                  shape_param.data,
                  strides_param.data,
                  ndim);
            } else {
              hipLaunchKernelGGL(
                  (rocm::unary_g<Op, InType, OutType, int32_t, 4>),
                  dim3(num_blocks_x, num_blocks_y), block_dims, 0, encoder.stream(),
                  rocm::gpu_ptr<InType>(in),
                  rocm::gpu_ptr<OutType>(out),
                  static_cast<int32_t>(rest),
                  shape_param.data,
                  strides_param.data,
                  ndim);
            }
          } else {
            if (large) {
              hipLaunchKernelGGL(
                  (rocm::unary_g<Op, InType, OutType, int64_t, 1>),
                  dim3(num_blocks_x, num_blocks_y), block_dims, 0, encoder.stream(),
                  rocm::gpu_ptr<InType>(in),
                  rocm::gpu_ptr<OutType>(out),
                  static_cast<int64_t>(rest),
                  shape_param.data,
                  strides_param.data,
                  ndim);
            } else {
              hipLaunchKernelGGL(
                  (rocm::unary_g<Op, InType, OutType, int32_t, 1>),
                  dim3(num_blocks_x, num_blocks_y), block_dims, 0, encoder.stream(),
                  rocm::gpu_ptr<InType>(in),
                  rocm::gpu_ptr<OutType>(out),
                  static_cast<int32_t>(rest),
                  shape_param.data,
                  strides_param.data,
                  ndim);
            }
          }
        }
      } else {
        throw std::runtime_error(
            std::string("Cannot do unary op ") + op + " on input of " +
            dtype_to_string(in.dtype()) + " with output of " +
            dtype_to_string(out.dtype()));
      }
    });
  });
}

template <typename Op>
void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& encoder = rocm::get_command_encoder(s);
  set_unary_output_data(
      inputs[0], out, [&](auto n) { return rocm::malloc_async(n, encoder); });
  unary_op_gpu_inplace<Op>(inputs, out, op, s);
}

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    roctxRangePush(#func "::eval_gpu");                               \
    auto& s = out.primitive().stream();                               \
    unary_op_gpu<rocm::func>(inputs, out, name(), s);                 \
    roctxRangePop();                                                  \
  }

} // namespace mlx::core

