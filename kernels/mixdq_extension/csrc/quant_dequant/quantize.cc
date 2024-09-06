#include <torch/extension.h>

#include "quant.h"
#include "quantization_kernels.h"

namespace mixdq {

// implemented in quantize_kernel.cu
torch::Tensor quantize_per_tensor_to_int8(torch::Tensor input, 
                                          torch::Tensor scale_inv,
                                          torch::Tensor zero_point)
{
    TORCH_CHECK(input.device().is_cuda(), "input should be on CUDA");
    TORCH_CHECK(input.device() == scale_inv.device(),
                "input and scale should be on the same device");
    TORCH_CHECK(input.device() == zero_point.device(),
                "input and zero_point should be on the same device");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input should be fp16");
    TORCH_CHECK(scale_inv.dtype() == torch::kFloat32, 
                "scale_inv should be fp32");
    TORCH_CHECK(zero_point.dtype() == torch::kFloat32, 
               "zero_point should be fp32");

    auto options = torch::TensorOptions().dtype(torch::kInt8)
                                         .device(input.device());
    torch::Tensor output = torch::empty_like(input, options);
    quantize_to_int8(input, scale_inv, zero_point, output);

    return output;
}

torch::Tensor quantize_per_tensor_to_int8_vectorized(torch::Tensor input, 
                                          torch::Tensor scale_inv,
                                          torch::Tensor zero_point)
{
    TORCH_CHECK(input.device().is_cuda(), "input should be on CUDA");
    TORCH_CHECK(input.device() == scale_inv.device(),
                "input and scale should be on the same device");
    TORCH_CHECK(input.device() == zero_point.device(),
                "input and zero_point should be on the same device");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input should be fp16");
    TORCH_CHECK(scale_inv.dtype() == torch::kFloat32, 
                "scale_inv should be fp32");
    TORCH_CHECK(zero_point.dtype() == torch::kFloat32, 
               "zero_point should be fp32");

    auto options = torch::TensorOptions().dtype(torch::kInt8)
                                         .device(input.device());
    torch::Tensor output = torch::empty_like(input, options);
    quantize_to_int8_vectorized(input, scale_inv, zero_point, output);

    return output;
}

void initQuantizationBindings(py::module m) {
  m.def("quantize_per_tensor_to_int8", 
        &quantize_per_tensor_to_int8, 
        "Quantize to int 8 per tensor with scale and zero point.");
  m.def("quantize_per_tensor_to_int8_vectorized", 
        &quantize_per_tensor_to_int8_vectorized, 
        "Quantize to int 8 per tensor with scale and zero point, use vectorized memory access to reduce load time");
}

} // namespace mixdq