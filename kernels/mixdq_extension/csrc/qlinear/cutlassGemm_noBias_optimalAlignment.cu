#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/util/device_memory.h"

#include "gemm_universal_with_visitor.h"
#include "cutlass_gemm_kernels.h"

namespace mixdq {
namespace {

// define basic configurations
using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
static const int MainLoopStages = 10;
static const int EpilogueStages = 2;
using ElementC = cutlass::half_t;
static const int AlignmentInput = 128 / cutlass::sizeof_bits<int8_t>::value;
static const int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementC>::value;

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
    OutputTileThreadMap, float,
    cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
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
    OutputTileThreadMap, float,
    cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
>;

using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<
    Compute1,
    EVTCompute0,
    Scale>;

using D = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC, 
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>
>;

using EVTNoBias = cutlass::epilogue::threadblock::Sm80EVT<
    D,
    EVTCompute1>;

// define the kernel
using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    // input (tensor A) configurations
    int8_t, 
    cutlass::layout::RowMajor, 
    cutlass::ComplexTransform::kNone, 
    AlignmentInput,
    // weight (tensor B) configurations
    int8_t, 
    cutlass::layout::ColumnMajor, 
    cutlass::ComplexTransform::kNone, 
    AlignmentInput,
    // output (tensor D) configurations
    ElementC, 
    cutlass::layout::RowMajor, 
    AlignmentOutput,
    int32_t,
    // Other configurations
    float, // element epilogue compute
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EVTNoBias,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    MainLoopStages,
    cutlass::arch::OpMultiplyAddSaturate,
    EpilogueStages
>::GemmKernel;

using DeviceOp = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

typename EVTNoBias::Arguments get_callback_args(
    const int ldD,
    const float* Bias0_data,
    const float* Scale_data,
    const ElementC* Bias1_data, // unused
    ElementC* D_data
) {
    EVTNoBias::Arguments callback_args{            
        {   
            {
                {},                                                             // Accum
                {Bias0_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},   // Bias0
                {}                                                              // Compute0
            },                                                                  // EVTCompute0
            {Scale_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},       // Scale
            {}                                                                  // Compute1
        },                                                                      // EVTCompute1
        {D_data, {ldD, cute::_1{}, cute::_0{}} }                                  // D
    };
    return callback_args;
}

typename DeviceOp::Arguments args_from_options(
    const int M, 
    const int N,
    const int K,
    const typename DeviceOp::ElementA* A_data,
    const typename DeviceOp::ElementB* B_data,
    const float* Bias0_data,
    const float* Scale_data,
    const typename DeviceOp::ElementC* Bias1_data,
    typename DeviceOp::ElementC* D_data
) {

    typename DeviceOp::GemmKernel::Epilogue::FusionCallbacks::Arguments 
    callback_args = get_callback_args(
        /*ldD=*/N, Bias0_data, Scale_data, Bias1_data, D_data
    );
    
    return typename DeviceOp::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,    // universal mode
        cutlass::gemm::GemmCoord({M, N, K}),        // problem size
        1,                                          // split k factor
        callback_args,                              // argument of EVT callbacks
        A_data, B_data, nullptr, nullptr,           // A/B/C/D pointers (C/D unused)
        M*K, N*K, 0, 0,                             // A/B/C/D batch stride (C/D unused)
        K, K, 0, 0                                  // A/B/C/D stride (C/D unused)
    );
}

// CUDA Kernel implementation
cutlass::Status qlinear_kernel_run(int M, int N, int K,
    const typename DeviceOp::ElementA* A, 
    const typename DeviceOp::ElementB* B, 
    const float* Bias0_data,
    const float* Scale_data,
    const typename DeviceOp::ElementC* Bias1_data,
    typename DeviceOp::ElementC* D,
    c10::Device device) 
{       
    DeviceOp gemm_op;

    typename DeviceOp::Arguments arguments = args_from_options(
        M, N, K, A, B, Bias0_data, Scale_data, Bias1_data, D);
    
    size_t workspace_size = DeviceOp::get_workspace_size(arguments);

    torch::TensorOptions options = torch::TensorOptions()
                                   .dtype(at::kByte)
                                   .device(device);
    at::Tensor workspace = torch::empty({static_cast<int>(workspace_size)}, 
                                        options);
    void* workspace_ptr = reinterpret_cast<void*>(workspace.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cutlass::Status status = gemm_op.initialize(arguments,
                                                workspace_ptr,
                                                stream);     // CUDA stream

    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    status = gemm_op(stream);
    return status;
}

}   // namespace {}

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
) {
    return qlinear_kernel_run(
        M, N, K, 
        reinterpret_cast<typename DeviceOp::ElementA*>(A.data_ptr()),
        reinterpret_cast<typename DeviceOp::ElementB*>(B.data_ptr()),
        reinterpret_cast<float*>(bias0.data_ptr()),
        reinterpret_cast<float*>(scale.data_ptr()),
        nullptr, // unused
        reinterpret_cast<typename DeviceOp::ElementC*>(D.data_ptr()),
        A.device()
        );
}

}   // namespace mixdq
