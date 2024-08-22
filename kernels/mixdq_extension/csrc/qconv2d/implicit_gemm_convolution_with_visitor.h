
#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

#include "cutlass/conv/kernel/implicit_gemm_convolution.h"

namespace cutlass {
namespace conv {
namespace kernel {

template<
    typename Mma_,
    typename Epilogue_,
    typename ThreadblockSwizzle_,
    conv::Operator ConvOperator,
    ///! Convolutional operator on 2D or 3D problem
    typename ConvProblemSize_ = Conv2dProblemSize,  
    ///! Group mode
    conv::GroupMode GroupMode_ = conv::GroupMode::kNone    
>
class ImplicitGemmConvolutionWithEpilogueVisitor {
public:
    
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Layout;
    using LayoutC = LayoutA;

    using WarpMmaOperator = typename Mma::Policy::Operator;

    using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
    using MathOperator = typename ArchMmaOperator::Operator;
    
    using OperatorClass = typename WarpMmaOperator::OperatorClass;
    using ArchTag = typename WarpMmaOperator::ArchTag;

    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename WarpMmaOperator::Shape;
    using InstructionShape = typename ArchMmaOperator::Shape;

    static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK; 

    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
    using ElementCompute = float;

    static int const kStages = Mma::kStages;
    static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    using TensorRefA = typename Mma::IteratorA::TensorRef;
    using TensorRefB = typename Mma::IteratorB::TensorRef;
    using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

    /// Conv dimension and problem size structure (Conv2d or Conv3d)
    using ConvProblemSize = ConvProblemSize_;

    static conv::GroupMode const kGroupMode = GroupMode_;
    static Operator const kConvolutionalOperator = ConvOperator;
    static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm; 

    using FusionCallbacks = typename Epilogue::FusionCallbacks;
    
    /// Argument structure
    struct Arguments {

        //
        // Data members
        //

        ConvProblemSize problem_size;
        TensorRefA ref_A;
        TensorRefB ref_B;
        TensorRefC ref_C;
        TensorRefC ref_D;
        typename EpilogueOutputOp::Params output_op;
        SplitKMode split_k_mode;

        //
        // Methods
        //

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments() { }
    
        CUTLASS_HOST_DEVICE 
        Arguments(
        ConvProblemSize const & problem_size
        ):
        problem_size(problem_size) { }

        CUTLASS_HOST_DEVICE
        Arguments(
        ConvProblemSize const & problem_size,
        TensorRefA const & ref_A,
        TensorRefB const & ref_B,
        TensorRefC const & ref_C,
        TensorRefC const & ref_D,
        typename EpilogueOutputOp::Params const & output_op,
        SplitKMode const & split_k_mode = SplitKMode::kSerial
        ):
        problem_size(problem_size),
        ref_A(ref_A),
        ref_B(ref_B),
        ref_C(ref_C),
        ref_D(ref_D),
        output_op(output_op),
        split_k_mode(split_k_mode)
        {

        }

    };

    /// Parameters structure
    struct Params {
        ConvProblemSize problem_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        gemm::GemmCoord implicit_gemm_problem_size;
        cute::Shape<int32_t,int32_t,int32_t> implicit_gemm_problem_shape;
        int swizzle_log_tile;
        int gemm_k_iterations;
        int gemm_k_iterations_per_channel;
        typename Mma::IteratorA::Params iterator_A;
        typename Mma::IteratorA::Element const *ptr_A;
        typename Mma::IteratorB::Params iterator_B;
        typename Mma::IteratorB::Element const *ptr_B;
        typename FusionCallbacks::Params output_op;
        int *semaphore;
        SplitKMode split_k_mode;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params(): swizzle_log_tile(0), gemm_k_iterations(0) { }

        /// 
        CUTLASS_HOST_DEVICE
        Params(
            Arguments const &args,
            int *semaphore = nullptr
        ):
            problem_size(args.problem_size),
            implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
            implicit_gemm_problem_shape({implicit_gemm_problem_size.m(), implicit_gemm_problem_size.n(), 1}),
            iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
            ptr_A(args.ref_A.data()),
            iterator_B(args.problem_size, args.ref_B.layout()),
            ptr_B(args.ref_B.data()),
            output_op(FusionCallbacks::to_underlying_arguments(implicit_gemm_problem_shape, args.output_op, nullptr /*workspace*/)),
            semaphore(semaphore),
            split_k_mode(args.split_k_mode)
        {
            gemm_k_iterations = implicit_gemm_k_iterations(
                kConvolutionalOperator,
                ThreadblockShape::kK,
                args.problem_size,
                kIteratorAlgorithm,
                kGroupMode,
                ThreadblockShape::kN);

            gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
                kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

            ThreadblockSwizzle threadblock_swizzle;

            grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
                implicit_gemm_problem_size,
                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
                args.problem_size.split_k_slices);

            swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
        }
    };

    /// Shared memory storage structure
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };


    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    ImplicitGemmConvolutionWithEpilogueVisitor() { } 


    /// Executes one ImplicitGEMM
    CUTLASS_DEVICE
    void operator()(Params const &params, SharedStorage &shared_storage) {

        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_idx =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if CTA is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

        return;
        }

        // Compute position within threadblock
        int thread_idx = threadIdx.x;
        int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;
        if (kGroupMode != GroupMode::kNone) {
        if (kGroupMode != GroupMode::kDepthwise) {
            int k_per_group = params.problem_size.K / params.problem_size.groups;
            int group_idx = threadblock_tile_idx.n() * Mma::Shape::kN / k_per_group;
            int channels_per_group = params.problem_size.C / params.problem_size.groups;
            iterator_A_column_offset += group_idx * channels_per_group;
        } else {
            iterator_A_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
        }
        } 

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(
        params.iterator_A,
        params.problem_size,
        params.ptr_A,
        thread_idx,
        MatrixCoord(
            threadblock_tile_idx.m() * Mma::Shape::kM,
            iterator_A_column_offset
        )
        );
        
        typename Mma::IteratorB iterator_B(
        params.iterator_B,
        params.problem_size,
        params.ptr_B,
        thread_idx,
        MatrixCoord(
            threadblock_tile_idx.k() * Mma::Shape::kK,
            threadblock_tile_idx.n() * Mma::Shape::kN
        )
        );

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = canonical_warp_idx_sync();
        int lane_idx = threadIdx.x % 32;

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Compute threadblock-scoped matrix multiply-add
        mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, params.gemm_k_iterations_per_channel);

        //
        // Epilogue
        //

        threadblock_tile_idx = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        Epilogue epilogue(
            params.output_op,
            shared_storage.epilogue, 
            thread_idx, 
            warp_idx, 
            lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(accumulators, threadblock_tile_idx, params.implicit_gemm_problem_shape, thread_idx); 
    } 

};


}   // namespace kernel
}   // namespace conv
}   // namespace cutlass