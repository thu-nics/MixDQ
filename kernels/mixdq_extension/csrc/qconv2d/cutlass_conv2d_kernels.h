#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>
#include "cutlass/cutlass.h"

namespace mixdq {

cutlass::Status
cutlass_tensorop_f16_i16832conv2dfprop_no_padding_small_alignment(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias0,
    at::Tensor scale,
    at::optional<at::Tensor> bias1,
    at::Tensor& output
);

cutlass::Status
cutlass_tensorop_f16_i16832conv2dfprop_no_padding_optimal_alignment(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias0,
    at::Tensor scale,
    at::optional<at::Tensor> bias1,
    at::Tensor& output
);

cutlass::Status
cutlass_tensorop_f16_i16832conv2dfprop_with_padding_small_alignment(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias0,
    at::Tensor scale,
    at::optional<at::Tensor> bias1,
    at::Tensor& output
);

cutlass::Status
cutlass_tensorop_f16_i16832conv2dfprop_with_padding_optimal_alignment(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias0,
    at::Tensor scale,
    at::optional<at::Tensor> bias1,
    at::Tensor& output
);

}