#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"

#include "default_conv2d_fprop_with_visitor.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "../common/default_epilogue_tensor_op.h"
#include "cutlass/util/device_memory.h"

#include "cutlass_conv2d_kernels.h"

namespace mixdq {
namespace {

// define basic configurations
using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
static const int MainLoopStages = 6;
static const int EpilogueStages = 2;
using ElementC = cutlass::half_t;
static const int AlignmentInput = 4;
static const int AlignmentOutput = 4;
static const cutlass::conv::IteratorAlgorithm IteratorAlgorithm = 
    cutlass::conv::IteratorAlgorithm::kOptimized;

// define epilogue based on: D = (Accum - Bias0) * Scale [+ Bias1]
using \
OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape,
    WarpShape,
    ElementC,
    AlignmentOutput,
    EpilogueStages
>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

using Bias0 = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>
>;

using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::minus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
    Compute0, 
    Accum,
    Bias0
>;

using Scale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>
>;

using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<
    Compute1,
    EVTCompute0,
    Scale>;

using Bias1 = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, cutlass::half_t,
    cute::Stride<cute::_0, cute::_1, cute::_0>
>;

using Compute2 = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute2 = cutlass::epilogue::threadblock::Sm80EVT<
    Compute2,
    EVTCompute1,
    Bias1>;

using D = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, cutlass::half_t, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::_1, cute::_0>
>;

using EVTDNoBias = cutlass::epilogue::threadblock::Sm80EVT<
    D,
    EVTCompute1>;

using EVTDWithBias = cutlass::epilogue::threadblock::Sm80EVT<
    D,
    EVTCompute2>;

template<typename EVT>
using KernelGenerator = 
typename cutlass::conv::kernel::DefaultConv2dFpropWithVisitor<
    int8_t, 
    cutlass::layout::TensorNHWC,
    int8_t, 
    cutlass::layout::TensorNHWC,
    cutlass::half_t, 
    cutlass::layout::TensorNHWC,
    int32_t,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89,
    ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EVT,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    MainLoopStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EpilogueStages,
    IteratorAlgorithm,
    cutlass::conv::StrideSupport::kStrided,
    AlignmentInput,
    AlignmentInput,
    AlignmentOutput
>::Kernel;

using KernelNoBias = KernelGenerator<EVTDNoBias>;
using KernelWithBias = KernelGenerator<EVTDWithBias>;
using DeviceOpNoBias = cutlass::conv::device::ImplicitGemmConvolution<KernelNoBias>;
using DeviceOpWithBias = cutlass::conv::device::ImplicitGemmConvolution<KernelWithBias>;

typename EVTDNoBias::Arguments get_callback_args(
    DeviceOpNoBias& gemm_op,
    const int ldD,
    const float* Bias0_data,
    const float* Scale_data,
    const ElementC* Bias1_data, // unused
    ElementC* D_data
) {
    typename EVTDNoBias::Arguments callback_args{            
        {   
            {
                {},                                                             // Accum
                {Bias0_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},   // Bias0
                {}                                                              // Compute0
            },                                                                  // EVTCompute0
            {Scale_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},       // Scale
            {}                                                                  // Compute1
        },                                                                      // EVTCompute1
        {D_data, {ldD, cute::_1{}, cute::_0{}} }                                // D
    };
    return callback_args;
}

typename EVTDWithBias::Arguments get_callback_args(
    DeviceOpWithBias& gemm_op,
    const int ldD,
    const float* Bias0_data,
    const float* Scale_data,
    const ElementC* Bias1_data,
    ElementC* D_data
) {
    typename EVTDWithBias::Arguments callback_args{
        {
            {   
                {
                    {},                                                             // Accum
                    {Bias0_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},   // Bias0
                    {}                                                              // Compute0
                },                                                                  // EVTCompute0
                {Scale_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},       // Scale
                {}                                                                  // Compute1
            },                                                                      // EVTCompute1
            {Bias1_data, ElementC(0), {cute::_0{}, cute::_1{}, cute::_0{}}},        // Bias1
            {}                                                                      // Compute2
        },                                                                          // EVTCompute2
        {D_data, {ldD, cute::_1{}, cute::_0{}} }                                    // D
    };
    return callback_args;
}

template<typename DeviceOp>
typename DeviceOp::Arguments args_from_options(
    DeviceOp& op,
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w,
    typename DeviceOp::ElementA* A_data,
    typename DeviceOp::ElementB* B_data,
    float* Bias0_data,
    const float* Scale_data,
    const typename DeviceOp::ElementC* Bias1_data, // unused
    typename DeviceOp::ElementC* D_data
) {
    cutlass::Tensor4DCoord input_size(N, H, W, C);
    cutlass::Tensor4DCoord filter_size(K, R, S, C);
    cutlass::Tensor4DCoord output_size(
        N,
        (H + padding_h*2 - R) / stride_h + 1,
        (W + padding_w*2 - S) / stride_w + 1,
        K
    );

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        cutlass::Tensor4DCoord(padding_h, padding_h, padding_w, padding_w),
        cutlass::MatrixCoord(stride_h, stride_w),
        cutlass::MatrixCoord(dilation_h, dilation_w),
        output_size,
        cutlass::conv::Mode::kCrossCorrelation,
        1 /* split_k_slices */
    );
    
    cutlass::layout::TensorNHWC layout_A(input_size.c(), 
        input_size.w()*input_size.c(), 
        input_size.h()*input_size.w()*input_size.c());
    cutlass::layout::TensorNHWC layout_B(filter_size.c(), 
        filter_size.w()*filter_size.c(), 
        filter_size.h()*filter_size.w()*filter_size.c());
    cutlass::layout::TensorNHWC layout_C(output_size.c(), 
        output_size.w()*output_size.c(), 
        output_size.h()*output_size.w()*output_size.c());
    
    typename DeviceOp::UnderlyingKernel::Epilogue::FusionCallbacks::Arguments
    callback_args = get_callback_args(
        op, /*ldD=*/K, Bias0_data, Scale_data, Bias1_data, D_data
    );

    return typename DeviceOp::Arguments(
        problem_size,
        typename DeviceOp::UnderlyingKernel::TensorRefA(A_data, layout_A),
        typename DeviceOp::UnderlyingKernel::TensorRefB(B_data, layout_B),
        typename DeviceOp::UnderlyingKernel::TensorRefC(nullptr, layout_C),
        typename DeviceOp::UnderlyingKernel::TensorRefC(nullptr, layout_C),
        callback_args
    );
}


// CUDA Kernel Implementation
template<typename DeviceOp>
cutlass::Status qconv2d_kernel_run(
    const int N, const int H, const int W, const int C, const int K,
    const int R, const int S, const int padding_h, const int padding_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, 
    typename DeviceOp::ElementA* A,
    typename DeviceOp::ElementB* B,
    float* Bias0,
    float* Scale,
    typename DeviceOp::ElementC* Bias1,
    typename DeviceOp::ElementC* D
) {
    DeviceOp conv2d_op;

    typename DeviceOp::Arguments arguments = args_from_options<DeviceOp>(
        conv2d_op, 
        N, H, W, C, K, R, S, padding_h, padding_w, stride_h, stride_w, 
        dilation_h, dilation_w, A, B, Bias0, Scale, Bias1, D);
    
    size_t workspace_size = DeviceOp::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cutlass::Status status = conv2d_op.initialize(arguments,
                                                workspace.get(),
                                                stream);     // CUDA stream

    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = conv2d_op(stream);
    return status;
}

} // namespace {}

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
) {
    if (bias1.has_value()) {
        return qconv2d_kernel_run<DeviceOpWithBias>(
            N, H, W, C, K, R, S, padding_h, padding_w, stride_h, stride_w, 
            dilation_h, dilation_w,
            reinterpret_cast<typename DeviceOpWithBias::ElementB*>(input.data_ptr()),
            reinterpret_cast<typename DeviceOpWithBias::ElementA*>(weight.data_ptr()),
            reinterpret_cast<float*>(bias0.data_ptr()),
            reinterpret_cast<float*>(scale.data_ptr()),
            reinterpret_cast<typename DeviceOpWithBias::ElementC*>(bias1.value().data_ptr()),
            reinterpret_cast<typename DeviceOpWithBias::ElementC*>(output.data_ptr())
        );
    }
    else {
        return qconv2d_kernel_run<DeviceOpNoBias>(
            N, H, W, C, K, R, S, padding_h, padding_w, stride_h, stride_w, 
            dilation_h, dilation_w,
            reinterpret_cast<typename DeviceOpNoBias::ElementB*>(input.data_ptr()),
            reinterpret_cast<typename DeviceOpNoBias::ElementA*>(weight.data_ptr()),
            reinterpret_cast<float*>(bias0.data_ptr()),
            reinterpret_cast<float*>(scale.data_ptr()),
            nullptr, // dummy
            reinterpret_cast<typename DeviceOpNoBias::ElementC*>(output.data_ptr())
        );
    }
}

} // namespace mixdq