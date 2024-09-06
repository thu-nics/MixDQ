#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <vector>

namespace mixdq {

namespace {

template<typename T>
__global__
void activation_zero_point_propagate_kernel(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int P, const int Q, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    const T* filter_sum_by_input_channels,
    const float* input_zp_ptr,
    float* output_ptr
) {
    int k = blockIdx.x;
    int filter_sum_offset = k * R * S;
    int nrol = N * P * Q;
    float zp = *input_zp_ptr;

    for (int rowId = threadIdx.x; rowId < nrol; rowId += blockDim.x)
    {
        int q = rowId % Q;
        int p = rowId / Q % P;
        int n = rowId / Q / P;

        int h0 = -1 * padding_h + p * stride_h;
        int w0 = -1 * padding_w + q * stride_w;

        T acc = 0;
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                int h = h0 + r * dilation_h;
                int w = w0 + s * dilation_w;
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    acc += filter_sum_by_input_channels[
                        s + r*S + filter_sum_offset
                    ];
                }
            }
        }
        float bias = float(acc) * zp;
        output_ptr[k + q*K + p*(Q*K) + n*(P*Q*K)] = bias;
    }
}

}   // namespace {}

at::Tensor activation_zero_point_propagate(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int P, const int Q, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    const at::Tensor& weight_sum_by_input_channel,
    const at::Tensor& input_zero_point
) {
    // create output tensor
    torch::TensorOptions options = torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(input_zero_point.device())
                                   .memory_format(at::MemoryFormat::ChannelsLast);
    at::Tensor output = torch::empty({N, K, P, Q}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // run kernel
    activation_zero_point_propagate_kernel<float><<</*griddim=*/K, /*blockdim=*/512, (size_t)0, stream>>>(
        N, H, W, C, K, R, S, P, Q, padding_h, padding_w, stride_h, stride_w, 
        dilation_h, dilation_w, 
        reinterpret_cast<float*>(weight_sum_by_input_channel.data_ptr()),
        reinterpret_cast<float*>(input_zero_point.data_ptr()),
        reinterpret_cast<float*>(output.data_ptr())
    );

    return output;
}

}
