#pragma once
#include <torch/extension.h>

namespace mixdq {

void quantize_to_int8(at::Tensor input, 
                      const at::Tensor scale_inv,
                      const at::Tensor zero_point,
                      at::Tensor output);

void quantize_to_int8_vectorized(at::Tensor input, 
                      const at::Tensor scale_inv,
                      const at::Tensor zero_point,
                      at::Tensor output);

} // namespace mixdq