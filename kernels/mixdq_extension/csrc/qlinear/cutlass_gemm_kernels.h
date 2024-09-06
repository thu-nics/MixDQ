#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>

#include "cutlass/cutlass.h"

namespace mixdq {

cutlass::Status
cutlass_tensorop_f16_i16832gemm_evt_no_bias_small_alignment(
    const int M, 
    const int N, 
    const int K,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor bias0,
    const at::Tensor scale,
    at::Tensor& D
);

cutlass::Status
cutlass_tensorop_f16_i16832gemm_evt_no_bias_optimal_alignment(
    const int M, 
    const int N, 
    const int K,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor bias0,
    const at::Tensor scale,
    at::Tensor& D
);

cutlass::Status
cutlass_tensorop_f16_i16832gemm_evt_with_bias_small_alignment(
    const int M, 
    const int N, 
    const int K,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor bias0,
    const at::Tensor scale,
    const at::Tensor bias1, // unused
    at::Tensor& D
);

cutlass::Status
cutlass_tensorop_f16_i16832gemm_evt_with_bias_optimal_alignment(
    const int M, 
    const int N, 
    const int K,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor bias0,
    const at::Tensor scale,
    const at::Tensor bias1, // unused
    at::Tensor& D
);


// debug only: FP16 cutlass reference
cutlass::Status
cutlass_tensorop_f16_i16832gemm_for_reference(
    const int M, 
    const int N, 
    const int K,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor bias1, // unused
    at::Tensor& D
);

}