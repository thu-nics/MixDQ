/*! \file
    \brief
    Default configuration for a Conv2d Fprop with epilogue visitor callbacks
    Refer to cutlass/gemm/kernel/default_gemm_universal_with_visitor.h
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

#include "implicit_gemm_convolution_with_visitor.h"

namespace cutlass {
namespace conv {
namespace kernel {

template<
    typename ElementA, 
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ElementEpilogueCompute,
    typename OperatorClass,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename FusionCallbacks,
    typename ThreadblockSwizzle,
    int Stages,
    typename MathOperatorTag,
    int EpilogueStages = 1,
    conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
    conv::StrideSupport StrideSupport = StrideSupport::kStrided,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value,
    /// Access granularity of D matrix in units of elements
    int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value
>
struct DefaultConv2dFpropWithVisitor {
    using Base = typename DefaultConv2dFprop<
        ElementA, 
        LayoutA, 
        ElementB, 
        LayoutB, 
        ElementC, 
        LayoutC, 
        ElementAccumulator,
        OperatorClass, 
        ArchTag, 
        ThreadblockShape, 
        WarpShape, 
        InstructionShape, 
        typename cutlass::epilogue::thread::LinearCombination<
            ElementC, 
            AlignmentC,
            ElementAccumulator,
            ElementEpilogueCompute
        >,
        ThreadblockSwizzle,
        Stages, 
        MathOperatorTag,
        IteratorAlgorithm,
        StrideSupport,
        AlignmentA,
        AlignmentB
    >::Kernel;

    using Mma = typename Base::Mma;
    static const int kPartitionsK = Base::kPartitionsK;

    using Epilogue = 
    cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
        typename Base::Epilogue,
        FusionCallbacks, 
        EpilogueStages
    >;
    
    using Kernel = 
    cutlass::conv::kernel::ImplicitGemmConvolutionWithEpilogueVisitor<
        Mma,
        Epilogue,
        ThreadblockSwizzle, 
        cutlass::conv::Operator::kFprop
    >;
};

}   // namespace kernel
}   // namespace conv
}   // namespace cutlass