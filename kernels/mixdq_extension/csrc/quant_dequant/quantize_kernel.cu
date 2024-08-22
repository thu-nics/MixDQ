#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include "quantization_kernels.h"

namespace mixdq {
namespace {

__global__ void quantize_to_int8_kernel
(int8_t *output, const __half* input, const float* scale_inv_ptr, 
const float* zero_point_ptr, int64_t numel)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int global_id = bid * blockDim.x + tid;
  float scale_inv_val = *scale_inv_ptr;
  float zero_point_val = *zero_point_ptr;
  
  for (int i = global_id; i < numel; i += gridDim.x *blockDim.x) {
    float x = __half2float(input[i]);
    int x_int = lrintf((x * scale_inv_val + zero_point_val));
    x_int = min(max(x_int, -128), 127);
    int8_t x_int8 = static_cast<int8_t>(x_int);
    output[i] = x_int8;
  }
}

}   // namespace {}


void quantize_to_int8(const at::Tensor input, 
                      const at::Tensor scale_inv,
                      const at::Tensor zero_point,
                      at::Tensor output)
{
    int64_t numel = input.numel();
    const int block_size = 256;  // original: 256
    int64_t grid_size = (numel + block_size - 1) / block_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    quantize_to_int8_kernel<<<grid_size, block_size, (size_t)0, stream>>>(
        reinterpret_cast<int8_t*>(output.data_ptr()),
        reinterpret_cast<__half*>(input.data_ptr()), 
        reinterpret_cast<float*>(scale_inv.data_ptr()),
        reinterpret_cast<float*>(zero_point.data_ptr()),
        numel
    );
}

} // namespace mixdq