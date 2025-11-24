// Copyright © 2025 Apple Inc.

#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace mlx::core {

#define NO_GPU_MULTI(func)                                             \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    throw std::runtime_error(#func " has no ROCm implementation.");    \
  }

#define NO_GPU_USE_FALLBACK(func)     \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  NO_GPU_MULTI(func)

#define NO_GPU(func)                                                  \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    throw std::runtime_error(#func " has no ROCm implementation.");   \
  }

// Operations not yet implemented for ROCm
NO_GPU(ArgPartition)
NO_GPU(ArgReduce)
NO_GPU(ArgSort)
NO_GPU(BlockMaskedMM)
NO_GPU(Convolution)
NO_GPU(FFT)
NO_GPU(Gather)
NO_GPU(GatherAxis)
NO_GPU(GatherMM)
NO_GPU(GatherQMM)
NO_GPU(Hadamard)
NO_GPU(Inverse)
NO_GPU(Cholesky)
NO_GPU(Load)
NO_GPU(LogSumExp)
NO_GPU(MaskedScatter)
NO_GPU(Partition)
NO_GPU(QuantizedMatmul)
NO_GPU(RandomBits)
NO_GPU(Scan)
NO_GPU(Scatter)
NO_GPU(ScatterAxis)
NO_GPU(Select)
NO_GPU(SegmentedMM)
NO_GPU(Softmax)
NO_GPU(Sort)

NO_GPU_MULTI(Compiled)
NO_GPU_MULTI(LUF)
NO_GPU_MULTI(QRF)
NO_GPU_MULTI(SVD)
NO_GPU_MULTI(Eig)
NO_GPU_MULTI(Eigh)

namespace fast {
NO_GPU_USE_FALLBACK(LayerNorm)
NO_GPU_MULTI(LayerNormVJP)
NO_GPU_USE_FALLBACK(RMSNorm)
NO_GPU_MULTI(RMSNormVJP)
NO_GPU_USE_FALLBACK(RoPE)
NO_GPU_MULTI(ScaledDotProductAttention)
NO_GPU_MULTI(ScaledDotProductAttentionVJP)
NO_GPU_MULTI(ConvertFP8)
NO_GPU_MULTI(Quantize)
NO_GPU_MULTI(CustomKernel)

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  return true;
}

bool ScaledDotProductAttentionVJP::use_fallback(
    const array& q,
    Stream s) {
  return true;
}
} // namespace fast

namespace distributed {
NO_GPU_MULTI(AllReduce)
NO_GPU_MULTI(AllGather)
NO_GPU_MULTI(Send)
NO_GPU_MULTI(Recv)
NO_GPU_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core

