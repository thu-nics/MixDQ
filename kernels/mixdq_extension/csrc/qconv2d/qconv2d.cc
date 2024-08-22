#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/ArrayRef.h>
#include <torch/torch.h>
#include <vector>

#include "qconv2d.h"
#include "cutlass_conv2d_kernels.h"

namespace mixdq {

// forward declaration
at::Tensor activation_zero_point_propagate(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int P, const int Q, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    const at::Tensor& weight_sum_by_input_channel,
    const at::Tensor& input_zero_point
);

// C++ function
at::Tensor
qconv2d_w8_a8_ohalf(at::Tensor input_int8,
                    at::Tensor weight_int8,
                    at::Tensor weight_scale,
                    at::Tensor input_scale,
                    at::Tensor input_zero_point,
                    at::Tensor scale,
                    at::optional<at::Tensor> weight_sum_by_input_channels,
                    at::optional<at::Tensor> bias0,
                    at::optional<const at::Tensor> bias,
                    at::optional<int> stride_,
                    at::optional<int> padding_,
                    at::optional<int> dilation_)
{
    // Sanity check of tensor device, types, and layout
    TORCH_CHECK(input_int8.device().is_cuda(), "Input should be on GPU.");
    TORCH_CHECK(input_int8.device() == weight_int8.device(), 
                "input and weight_int8 should be on the same device.");
    TORCH_CHECK(input_int8.device() == weight_scale.device(), 
                "input and weight_scale should be on the same device.");
    TORCH_CHECK(input_int8.device() == input_scale.device(), 
                "input and input_scale should be on the same device.");
    TORCH_CHECK(input_int8.device() == input_zero_point.device(), 
                "input and input_zero_point should be on the same device.");
    if (weight_sum_by_input_channels.has_value()) {
        TORCH_CHECK(
        input_int8.device() == weight_sum_by_input_channels.value().device(),
        "input and weight_sum_by_input_channels should be on the same device.");
    }
    if (bias0.has_value()) {
        TORCH_CHECK(
        input_int8.device() == bias0.value().device(),
        "input and bias0 should be on the same device.");
    }
    if (bias.has_value()) {
        TORCH_CHECK(input_int8.device() == bias.value().device(), 
                    "input and bias should be on the same device.");
    }

    TORCH_CHECK(input_int8.scalar_type() == torch::kInt8, 
                "input_int8 should be int8 type");
    TORCH_CHECK(weight_int8.scalar_type() == torch::kInt8, 
                "weight_int8 should be int8 type");
    TORCH_CHECK(weight_scale.scalar_type() == torch::kFloat32, 
                "Currently only support weight_scale with float32 type");
    TORCH_CHECK(input_scale.scalar_type() == torch::kFloat32, 
                "Currently only support input_scale with float32 type");
    TORCH_CHECK(input_zero_point.scalar_type() == torch::kFloat32, 
                "Currently only support input_zero_point with float32 type");
    if (weight_sum_by_input_channels.has_value()) {
        TORCH_CHECK(
        weight_sum_by_input_channels.value().scalar_type() == torch::kFloat32,
        "Currently only support weight_sum_by_input_channels with float32 type");
    }
    if (bias0.has_value()) {
        TORCH_CHECK(
        bias0.value().scalar_type() == torch::kFloat32,
        "Currently only support bias0 with float32 type");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat16, 
                    "Currently only support bias with float16 type");
    }

    at::Tensor input_int_channels_last = input_int8.contiguous(
        /*memory_format=*/at::MemoryFormat::ChannelsLast);

    at::Tensor weight_int_channels_last = weight_int8.contiguous(
        /*memory_format=*/at::MemoryFormat::ChannelsLast);

    int N, H, W, C;
    N = input_int8.size(0);
    C = input_int8.size(1);
    H = input_int8.size(2);
    W = input_int8.size(3);
    int K, R, S;
    K = weight_int8.size(0);
    R = weight_int8.size(2);
    S = weight_int8.size(3);
    int padding =0, stride =1, dilation =1;
    if (padding_.has_value()) { padding = padding_.value();}
    if (stride_.has_value()) { stride = stride_.value();}
    if (dilation_.has_value()) { dilation = dilation_.value();}
    
    int P, Q;
    P = (H + 2 * padding - dilation * (R -1) -1) / stride + 1;
    Q = (W + 2 * padding - dilation * (S -1) -1) / stride + 1;

    TORCH_CHECK(weight_scale.numel() == K, 
    "The size of the weight_scale vector should be equal to output_channels.");
    if (padding == 0) {
        TORCH_CHECK(bias0.value().numel() == K, 
        "The size of bias0 should equal output_channels.");
    }
    else {
        TORCH_CHECK(weight_sum_by_input_channels.value().numel() == K*R*S, 
        "The size of weight_sum_by_input_channels should equal K*R*S.");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().numel() == K,
        "The size of the bias vector should be equal to output_channels.");
    }

    // bias0
    at::Tensor bias0_;
    if (padding > 0) {
        bias0_ = activation_zero_point_propagate(
            N, H, W, C, K, R, S, P, Q, padding, padding, stride, stride, dilation,
            dilation, weight_sum_by_input_channels.value(), input_zero_point
        );
        // auto input_zp_broadcast = at::broadcast_to(input_zero_point, {N, 1, H, W});
        // bias0 = torch::nn::functional::conv2d(
        //     input_zp_broadcast.to(torch::kFloat32),
        //     weight_sum_by_input_channels,
        //     torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding)
        // ).contiguous(/*memory_format=*/at::MemoryFormat::ChannelsLast);
    }
    else {
        bias0_ = bias0.value();
    }

    // create output tensor

    torch::TensorOptions options = torch::TensorOptions()
                                   .dtype(torch::kHalf)
                                   .device(input_int8.device())
                                   .memory_format(at::MemoryFormat::ChannelsLast);

    at::Tensor D = torch::empty({N, K, P, Q}, options);

    cutlass::Status status;
    static const int alignment_A_optimal = 16;// 128/8bit
    static const int alignment_D_optimal = 8; // 128/16bit
    static const int alignment_A_small = 4; 
    static const int alignment_D_small = 4;

    if (C % alignment_A_optimal == 0 && K % alignment_D_optimal == 0) {
        if (padding > 0) {
            status = cutlass_tensorop_f16_i16832conv2dfprop_with_padding_optimal_alignment(
                N, H, W, C, K, R, S, padding, padding, stride, stride, dilation,
                dilation, input_int_channels_last, weight_int_channels_last, bias0_, 
                scale, bias, 
                D
            );
        }
        else {
            status = cutlass_tensorop_f16_i16832conv2dfprop_no_padding_optimal_alignment(
                N, H, W, C, K, R, S, padding, padding, stride, stride, dilation,
                dilation, input_int_channels_last, weight_int_channels_last, bias0_, 
                scale, bias, 
                D
            );
        }
    }
    else if (C % alignment_A_small == 0 && K % alignment_D_small == 0) {
        if (padding > 0) {
            status = cutlass_tensorop_f16_i16832conv2dfprop_with_padding_small_alignment(
                N, H, W, C, K, R, S, padding, padding, stride, stride, dilation,
                dilation, input_int_channels_last, weight_int_channels_last, bias0_, 
                scale, bias, 
                D
            );
        }
        else {
            status = cutlass_tensorop_f16_i16832conv2dfprop_no_padding_small_alignment(
                N, H, W, C, K, R, S, padding, padding, stride, stride, dilation,
                dilation, input_int_channels_last, weight_int_channels_last, bias0_, 
                scale, bias, 
                D
            );
        }
    }
    else {
        TORCH_CHECK(false, 
        "Int8 kernel with input or output alignment not to 4 is not supported.");
    }
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");

    return D;
}

void initQuantizedConv2dBindings(py::module m) {
    m.def("qconv2d_w8_a8_ohalf",
    py::overload_cast<const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      at::optional<at::Tensor>,
                      at::optional<at::Tensor>,
                      at::optional<const at::Tensor>,
                      at::optional<int>,
                      at::optional<int>,
                      at::optional<int>
                      >(&qconv2d_w8_a8_ohalf),
        py::arg("input_int8"),
        py::arg("weight_int8"),
        py::arg("weight_scale"),
        py::arg("input_scale"),
        py::arg("input_zero_point"),
        py::arg("scale"),
        py::arg("weight_sum_by_input_channels"),
        py::arg("bias0"),
        py::arg("bias") = nullptr,
        py::arg("stride") = 1,
        py::arg("padding") = 0,
        py::arg("dilation") = 1
        );
}

}   // namespace mixdq
