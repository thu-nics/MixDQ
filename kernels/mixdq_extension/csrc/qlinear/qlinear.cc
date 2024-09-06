#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>

#include <iostream> // for printing

#include "qlinear.h"
#include "cutlass_gemm_kernels.h"

namespace mixdq {

// C++ function
at::Tensor 
qlinear_w8_a8_ohalf(const at::Tensor input_int8,
                    const at::Tensor weight_int8,
                    const at::Tensor weight_scale,
                    const at::Tensor input_scale,
                    const at::Tensor input_zero_point,
                    const at::Tensor weight_sum_by_input_channels,
                    const at::Tensor scale,
                    const at::Tensor bias0,
                    at::optional<const at::Tensor> bias)
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
    TORCH_CHECK(input_int8.device() == weight_sum_by_input_channels.device(), 
                "input and input_zero_point should be on the same device.");
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
    TORCH_CHECK(weight_sum_by_input_channels.scalar_type() == torch::kFloat32, 
    "Currently only support weight_sum_by_input_channels with float32 type");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat16, 
                    "Currently only support bias with float16 type");
    }

    int output_channels, input_channels, num_samples;
    input_channels = weight_int8.size(1);
    output_channels = weight_int8.size(0);
    num_samples = input_int8.numel() / input_channels;
    
    TORCH_CHECK(weight_scale.numel() == output_channels, 
    "The size of the weight_scale vector should be equal to output_channels.");
    TORCH_CHECK(weight_sum_by_input_channels.numel() == output_channels, 
    "The size of weight_sum_by_input_channels should equal output_channels.");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().numel() == output_channels,
        "The size of the bias vector should be equal to output_channels.");
    }
    TORCH_CHECK(input_int8.size(-1) == input_channels, 
    "The last dimension of input and weight should match, got %d and %d.", 
    input_int8.size(-1), input_channels);
    
    at::Tensor input_int_contiguous = input_int8.contiguous();

    at::Tensor weight_int_contiguous = weight_int8.contiguous();

    // create output tensor

    std::vector<int64_t> output_size;
    int ndim = input_int8.dim();
    for (int i = 0; i < ndim - 1; i++) {
        output_size.push_back(input_int8.size(i));
    } 
    output_size.push_back(output_channels);
    torch::IntArrayRef output_size_(output_size);

    at::Tensor D = torch::empty(output_size_, torch::TensorOptions()
                                              .dtype(torch::kHalf)
                                              .device(input_int8.device()));


    cutlass::Status status;
    
    static const int alignment_A_optimal = 16;// 128/8bit
    static const int alignment_D_optimal = 8; // 128/16bit
    static const int alignment_A_small = 4; 
    static const int alignment_D_small = 4;
    int M = num_samples, N = output_channels, K = input_channels;

    if (K % alignment_A_optimal == 0 && N % alignment_D_optimal == 0) {
        if (bias.has_value()) {
            status = cutlass_tensorop_f16_i16832gemm_evt_with_bias_optimal_alignment(
                M, N, K, input_int_contiguous, weight_int_contiguous, bias0, scale, 
                bias.value(), D
            );
        }
        else {
            status = cutlass_tensorop_f16_i16832gemm_evt_no_bias_optimal_alignment(
                M, N, K, input_int_contiguous, weight_int_contiguous, bias0, scale, 
                D
            );
        }
    }
    else if (K % alignment_A_small == 0 && N % alignment_D_small == 0) {
        if (bias.has_value()) {
            status = cutlass_tensorop_f16_i16832gemm_evt_with_bias_small_alignment(
                M, N, K, input_int_contiguous, weight_int_contiguous, bias0, scale, 
                bias.value(), D
            );
        }
        else {
            status = cutlass_tensorop_f16_i16832gemm_evt_no_bias_small_alignment(
                M, N, K, input_int_contiguous, weight_int_contiguous, bias0, scale, 
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

// C++ function
at::Tensor 
qlinear_fp_reference(const at::Tensor input,
                  const at::Tensor weight,
                  at::optional<const at::Tensor> bias)
{
    // Sanity check of tensor device, types, and layout
    TORCH_CHECK(input.scalar_type() == torch::kFloat16, 
                "input should be int8 type");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat16, 
                "weight should be int8 type");

    if (bias.has_value()) {
        TORCH_CHECK(input.device() == bias.value().device(), 
                    "input and bias should be on the same device.");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().scalar_type() == torch::kFloat16, 
                    "Currently only support bias with float16 type");
    }

    int output_channels, input_channels, num_samples;
    input_channels = weight.size(0);  // INFO: different from existing kenrel, we use RowMajor for Data_B
    output_channels = weight.size(1);
    num_samples = input.numel() / input_channels;

    std::cout << " Weight Size: " << weight.size(0) << ", " << weight.size(1) << std::endl;
    std::cout << " Input Size: " << input.size(0) << ", " << input.size(1) << std::endl;
    
    at::Tensor input_contiguous = input.contiguous();
    at::Tensor weight_contiguous = weight.contiguous();

    // create output tensor
    std::vector<int64_t> output_size;
    int ndim = input.dim();
    for (int i = 0; i < ndim - 1; i++) {
        output_size.push_back(input.size(i));
    } 
    output_size.push_back(output_channels);
    torch::IntArrayRef output_size_(output_size);

    // std::cout << "[Debug] output_size: " << output_size << std::endl;
    // std::cout << "[Debug] ndim: " << ndim << std::endl;

    at::Tensor D = torch::empty(output_size_, torch::TensorOptions()
                                              .dtype(torch::kHalf)
                                              .device(input.device()));


    cutlass::Status status;
    
    static const int alignment_A_optimal = 16;// 128/8bit
    static const int alignment_D_optimal = 8; // 128/16bit
    static const int alignment_A_small = 4; 
    static const int alignment_D_small = 4;
    int M = num_samples, N = output_channels, K = input_channels;
    std::cout << " M: " << M  << " N: " << N << " K: " << K << std::endl;

    status = cutlass_tensorop_f16_i16832gemm_for_reference(
                M, N, K, input_contiguous, weight_contiguous, bias.value(), D
            );

    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS kernel failed");

    return D;
}


void initQuantizedLinearBindings(py::module m) {
    m.def("qlinear_w8_a8_ohalf",
    py::overload_cast<const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      const at::Tensor, 
                      at::optional<const at::Tensor>>(&qlinear_w8_a8_ohalf),
        py::arg("input_int8"),
        py::arg("weight_int8"),
        py::arg("weight_scale"),
        py::arg("input_scale"),
        py::arg("input_zero_point"),
        py::arg("weight_sum_by_input_channels"),
        py::arg("scale"),
        py::arg("bias0"),
        py::arg("bias") = nullptr);

    m.def("qlinear_fp_reference",
    py::overload_cast<const at::Tensor, 
                    const at::Tensor,
                    at::optional<const at::Tensor>>(&qlinear_fp_reference),
    py::arg("input"),
    py::arg("weight"),
    py::arg("bias") = nullptr); // bias not used
}

} // namespace mixdq