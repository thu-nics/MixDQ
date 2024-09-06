#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h" // to add GemmUniversal


// #include "cutlass/gemm/device/gemm_universal_adapter.h"
// #include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
// #include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
// #include "cutlass/util/device_memory.h"

// #include "gemm_universal_with_visitor.h"
#include "cutlass_gemm_kernels.h"

namespace mixdq {
namespace {

// define basic configurations
// using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 128>;
// using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
// static const int MainLoopStages = 5;
// static const int EpilogueStages = 2;
// using ElementC = cutlass::half_t;
// static const int AlignmentInput = 128 / cutlass::sizeof_bits<int8_t>::value;
// static const int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementC>::value;

// // define epilogue based on: D = (Accum - Bias0) * Scale [+ Bias1]
// using \
// OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
//     ThreadblockShape,
//     WarpShape,
//     ElementC,
//     AlignmentOutput,
//     EpilogueStages
// >;

// using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

// using Bias0 = cutlass::epilogue::threadblock::VisitorRowBroadcast<
//     OutputTileThreadMap, float,
//     cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
// >;

// using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
//     cutlass::minus, float, float,
//     cutlass::FloatRoundStyle::round_to_nearest
// >;

// using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
//     Compute0, 
//     Accum,
//     Bias0
// >;

// using Scale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
//     OutputTileThreadMap, float,
//     cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
// >;

// using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
//     cutlass::multiplies, float, float,
//     cutlass::FloatRoundStyle::round_to_nearest
// >;

// using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<
//     Compute1,
//     EVTCompute0,
//     Scale>;

// using D = cutlass::epilogue::threadblock::VisitorAuxStore<
//     OutputTileThreadMap, ElementC, 
//     cutlass::FloatRoundStyle::round_to_nearest,
//     cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>
// >;

// using EVTNoBias = cutlass::epilogue::threadblock::Sm80EVT<
//     D,
//     EVTCompute1>;

// // define the kernel
// using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
//     // input (tensor A) configurations
//     int8_t, 
//     cutlass::layout::RowMajor, 
//     cutlass::ComplexTransform::kNone, 
//     AlignmentInput,
//     // weight (tensor B) configurations
//     int8_t, 
//     cutlass::layout::ColumnMajor, 
//     cutlass::ComplexTransform::kNone, 
//     AlignmentInput,
//     // output (tensor D) configurations
//     ElementC, 
//     cutlass::layout::RowMajor, 
//     AlignmentOutput,
//     int32_t,
//     // Other configurations
//     float, // element epilogue compute
//     cutlass::arch::OpClassTensorOp,
//     cutlass::arch::Sm80,
//     ThreadblockShape,
//     WarpShape,
//     cutlass::gemm::GemmShape<16, 8, 32>,
//     EVTNoBias,
//     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//     MainLoopStages,
//     cutlass::arch::OpMultiplyAddSaturate,
//     EpilogueStages
// >::GemmKernel;

// ------------------------------------------------------------
// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                      // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator  = cutlass::half_t;                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag
using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 32>;   // Threadblock-level tile size (concept: GemmShape)
using WarpShape           = cutlass::gemm::GemmShape<64, 64, 32>;     // Warp-level tile size (concept: GemmShape)
using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;      // Instruction-level tile size (concept: GemmShape)
constexpr int NumStages   = 4;                                        // Number of global->shared pipeline stages used in the GEMM mainloop

// Epilogue output operator
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,               // Element type for C and D matrix operands
    AlignmentC,             // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator,     // Element type from internal accumaccumulation
    ElementAccumulator>;    // Data type used to compute linear combination

// Classic data-parallel device GEMM implementation type
using DeviceOp = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

// using DeviceOp = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;


// typename EVTNoBias::Arguments get_callback_args(
//     const int ldD,
//     const float* Bias0_data,
//     const float* Scale_data,
//     const ElementC* Bias1_data, // unused
//     ElementC* D_data
// ) {
//     EVTNoBias::Arguments callback_args{            
//         {   
//             {
//                 {},                                                             // Accum
//                 {Bias0_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},   // Bias0
//                 {}                                                              // Compute0
//             },                                                                  // EVTCompute0
//             {Scale_data, float(0), {cute::_0{}, cute::_1{}, cute::_0{}}},       // Scale
//             {}                                                                  // Compute1
//         },                                                                      // EVTCompute1
//         {D_data, {ldD, cute::_1{}, cute::_0{}} }                                  // D
//     };
//     return callback_args;
// }

typename DeviceOp::Arguments args_from_options(
    const int M, 
    const int N,
    const int K,
    const typename DeviceOp::ElementA* A_data,
    const typename DeviceOp::ElementB* B_data,
    // const float* Bias0_data,
    // const float* Scale_data,
    const typename DeviceOp::ElementC* Bias1_data,
    typename DeviceOp::ElementC* D_data
) {

    // typename DeviceOp::GemmKernel::Epilogue::FusionCallbacks::Arguments 
    // callback_args = get_callback_args(
    //     /*ldD=*/N, Bias0_data, Scale_data, Bias1_data, D_data
    // );
    
    return typename DeviceOp::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,    // universal mode
        cutlass::gemm::GemmCoord({M, N, K}),        // problem size
        1,                                          // split k factor
        {                                           // epilogue parameters
        ElementAccumulator(1.0f),
        ElementAccumulator(0.0f)
        },
        // callback_args,                           // argument of EVT callbacks
        A_data, B_data, nullptr, D_data,            // A/B/C/D pointers (C/D unused)
        M*K, N*K, 0, M*N,                           // A/B/C/D batch stride (C/D unused)
        K, N, 0, N                                  // A/B/C/D stride (C/D unused)
    );
}

// CUDA Kernel implementation
cutlass::Status qlinear_kernel_run(int M, int N, int K,
    const typename DeviceOp::ElementA* A, 
    const typename DeviceOp::ElementB* B, 
    // const float* Bias0_data,
    // const float* Scale_data,
    const typename DeviceOp::ElementC* Bias1_data,
    typename DeviceOp::ElementC* D,
    c10::Device device) 
{       
    DeviceOp gemm_op;

    typename DeviceOp::Arguments arguments = args_from_options(
        M, N, K, A, B, Bias1_data, D);
    
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
cutlass_tensorop_f16_i16832gemm_for_reference(
    const int M, 
    const int N, 
    const int K,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor bias1,  //nullptr in, unused
    // const at::Tensor bias0,
    // const at::Tensor scale,
    at::Tensor& D
) {
    return qlinear_kernel_run(
        M, N, K, 
        reinterpret_cast<typename DeviceOp::ElementA*>(A.data_ptr()),
        reinterpret_cast<typename DeviceOp::ElementB*>(B.data_ptr()),
        // reinterpret_cast<float*>(bias0.data_ptr()),
        // reinterpret_cast<float*>(scale.data_ptr()),
        reinterpret_cast<typename DeviceOp::ElementC*>(bias1.data_ptr()),
        reinterpret_cast<typename DeviceOp::ElementC*>(D.data_ptr()),
        A.device()
        );
}

}   // namespace mixdq
