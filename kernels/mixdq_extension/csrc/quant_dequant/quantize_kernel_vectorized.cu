#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "quantization_kernels.h"
#include <stdio.h>


// #define CAST_TO_FLOAT4(value) (reinterpret_cast<float4>(&(value))[0])
// #define CAST_TO_INT8_4(value) (reinterpret_cast<char4>(&(value))[0])

#define CAST_TO_HALF4(pointer)   (reinterpret_cast<const half4 *>(&(pointer))[0])
#define CAST_TO_FLOAT4(pointer)  (reinterpret_cast<float4 *>(&(pointer))[0])
#define CAST_TO_INT8_4(pointer)  (reinterpret_cast<int8_4 *>(&(pointer))[0])

namespace mixdq {
namespace {

typedef struct __device_builtin__ __align__(4) int8_4 {
    int8_t x, y, z, w;
} int8_4;

typedef struct __device_builtin__ __align__(8) half4 {
    __half x, y, z, w;
} half4;


__global__ void quantize_to_int8_vectorized_kernel(int8_t *output, const __half* input, const float* scale_inv_ptr, const float* zero_point_ptr, int64_t numel) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    float scale_inv_val = *scale_inv_ptr;
    float zero_point_val = *zero_point_ptr;

    // printf("Global ID: %d\n", global_id);

    for (int i = global_id*4; i < numel; i += gridDim.x * blockDim.x * 4) {
        // Load 4 half elements into a float4
        half4 x_half = CAST_TO_HALF4(input[i]);
        float4 x;
        x.x = __half2float(x_half.x);
        x.y = __half2float(x_half.y);
        x.z = __half2float(x_half.z);
        x.w = __half2float(x_half.w);

        // printf("Finished input loading\n");

        // Quantize each element
        int4 x_int;
        x_int.x = lrintf((x.x * scale_inv_val + zero_point_val));
        x_int.y = lrintf((x.y * scale_inv_val + zero_point_val));
        x_int.z = lrintf((x.z * scale_inv_val + zero_point_val));
        x_int.w = lrintf((x.w * scale_inv_val + zero_point_val));

        // Clamp to int8 range
        x_int.x = min(max(x_int.x, -128), 127);
        x_int.y = min(max(x_int.y, -128), 127);
        x_int.z = min(max(x_int.z, -128), 127);
        x_int.w = min(max(x_int.w, -128), 127);

        // Store as int8_t
        int8_4 x_int8;
        x_int8.x = static_cast<int8_t>(x_int.x);
        x_int8.y = static_cast<int8_t>(x_int.y);
        x_int8.z = static_cast<int8_t>(x_int.z);
        x_int8.w = static_cast<int8_t>(x_int.w);

        // Store the results
        CAST_TO_INT8_4(output[i]) = x_int8;
    }
}

}   // namespace {}


void quantize_to_int8_vectorized(const at::Tensor input, 
                      const at::Tensor scale_inv,
                      const at::Tensor zero_point,
                      at::Tensor output)
{
    int64_t numel = input.numel();
    const int block_size = 256;  // original: 256
    int64_t grid_size = (numel/4 + block_size - 1) / block_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // printf("Grid Size: %d, Block Size: %d\n", grid_size, block_size);
    quantize_to_int8_vectorized_kernel<<<grid_size, block_size, (size_t)0, stream>>>(
        reinterpret_cast<int8_t*>(output.data_ptr()),
        reinterpret_cast<__half*>(input.data_ptr()), 
        reinterpret_cast<float*>(scale_inv.data_ptr()),
        reinterpret_cast<float*>(zero_point.data_ptr()),
        numel
    );
}

} // namespace mixdq