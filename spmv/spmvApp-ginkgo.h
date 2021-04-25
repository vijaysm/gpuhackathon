#ifndef __SPMVAPP_GINKGO_H__
#define __SPMVAPP_GINKGO_H__

#ifndef USE_GINKGO
#error "Define 'USE_GINKGO' and ensure makefile includes header path and libraries."
#endif

#include "spmvAppUtils.h"

// Ginkgo header files
#include <ginkgo/ginkgo.hpp>

// Define all supported SparseMatrix types in Ginkgo
typedef gko::matrix::Csr< MOABReal > GinkgoCSRMatrix;
typedef gko::matrix::Coo< MOABReal > GinkgoCOOMatrix;
typedef gko::matrix::Ell< MOABReal > GinkgoELLMatrix;
typedef gko::matrix::Hybrid< MOABReal > GinkgoHybridEllMatrix;
typedef gko::matrix::Sellp< MOABReal > GinkgoSellpMatrix;

template < typename MatrixType >
class GinkgoOperator : public SpMVOperator
{
  public:
    // Define a Vector type
    typedef gko::matrix::Dense< MOABReal > SpMV_VectorType;
    typedef MatrixType SpMV_MatrixType;
    typedef GinkgoCSRMatrix SpMV_DefaultMatrixType;

    GinkgoOperator( MOABSInt nOpRows, MOABSInt nOpCols, MOABSInt nOpNNZs, MOABSInt nVecs,
                    bool requireTransposeOp = false, const std::string exec_string = "omp" );
    virtual ~GinkgoOperator(){};  // TODO: clear operator memory
    void CreateOperator( const std::vector< MOABSInt >& vecRow, const std::vector< MOABSInt >& vecCol,
                         const std::vector< MOABReal >& vecS );
    virtual bool PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                      const std::vector< MOABReal >& vecAreasB );
    void PerformSpMV( int n_remap_iterations = 1 );
    void PerformSpMVTranspose( int n_remap_iterations = 1 );

    std::string matrix_type();

  private:
    void compute_transpose_operator( const std::vector< MOABSInt >& vecRow, const std::vector< MOABSInt >& vecCol,
                                     const std::vector< MOABReal >& vecS );
    void apply_operator( const std::unique_ptr< SpMV_VectorType >& rhs, std::unique_ptr< SpMV_VectorType >& result );
    void apply_transpose_operator( const std::unique_ptr< SpMV_VectorType >& rhs,
                                   std::unique_ptr< SpMV_VectorType >& result );

    std::string executor_string;
    std::shared_ptr< gko::Executor > device_executor;
    std::shared_ptr< gko::Executor > host_executor;

    std::unique_ptr< SpMV_MatrixType > mapOperator;
    std::unique_ptr< SpMV_MatrixType > mapTransposeOperator;
};

template < typename MatrixType >
GinkgoOperator< MatrixType >::GinkgoOperator( MOABSInt nRows, MOABSInt nCols, MOABSInt nNNZs, MOABSInt nRHSV,
                                              bool requireTransposeOp, const std::string exec_string )
    : SpMVOperator( nRows, nCols, nNNZs, nRHSV, requireTransposeOp ), executor_string( exec_string )
{
    // const auto executor_string = "omp";  // "reference", "omp", "cuda"
    // Figure out where to run the code
    std::map< std::string, std::function< std::shared_ptr< gko::Executor >() > > exec_map{
        { "omp", [] { return gko::OmpExecutor::create(); } },
        { "cuda", [] { return gko::CudaExecutor::create( 0, gko::OmpExecutor::create(), true ); } },
        { "reference", [] { return gko::ReferenceExecutor::create(); } }
    };

    // device_executor where Ginkgo will perform the computation
    this->device_executor = exec_map.at( executor_string )();  // throws if not valid

    // device_executor used by the application
    this->host_executor = this->device_executor->get_master();
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::compute_transpose_operator( const std::vector< MOABSInt >& vecRow,
                                                               const std::vector< MOABSInt >& vecCol,
                                                               const std::vector< MOABReal >& vecS )
{
    gko::matrix_data< MOABReal > mDataT( gko::dim< 2 >{ nOpCols, nOpRows } );
    auto& tripletTList = mDataT.nonzeros;
    tripletTList.reserve( vecS.size() );

    // loop over nnz and populate the sparse matrix operator
    for( size_t innz = 0; innz < vecS.size(); ++innz )
    {
        tripletTList.emplace_back( vecCol[innz] - 1, vecRow[innz] - 1, vecS[innz] );
    }

    mapTransposeOperator = MatrixType::create( device_executor );  // should this be "host_executor" ?
    // populate the CSR sparse matrix with the matrix_data object
    mapTransposeOperator->read( mDataT );
    return;
}

template <>
void GinkgoOperator< GinkgoCSRMatrix >::compute_transpose_operator( const std::vector< MOABSInt >& vecRow,
                                                                    const std::vector< MOABSInt >& vecCol,
                                                                    const std::vector< MOABReal >& vecS )
{
    // Nothing to do. We will use the transpose LinOp for CSR directly
}

template <>
void GinkgoOperator< GinkgoCOOMatrix >::compute_transpose_operator( const std::vector< MOABSInt >& vecRow,
                                                                    const std::vector< MOABSInt >& vecCol,
                                                                    const std::vector< MOABReal >& vecS )
{
    gko::matrix_data< MOABReal > mDataT( gko::dim< 2 >{ nOpCols, nOpRows } );
    auto& tripletTList = mDataT.nonzeros;
    tripletTList.reserve( vecS.size() );

    // loop over nnz and populate the sparse matrix operator
    for( size_t innz = 0; innz < vecS.size(); ++innz )
    {
        tripletTList.emplace_back( vecCol[innz] - 1, vecRow[innz] - 1, vecS[innz] );
    }

    mapTransposeOperator = GinkgoCOOMatrix::create( device_executor );  // should this be "host_executor" ?
    // populate the CSR sparse matrix with the matrix_data object
    mapTransposeOperator->read( mDataT );
    return;
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::CreateOperator( const std::vector< MOABSInt >& vecRow,
                                                   const std::vector< MOABSInt >& vecCol,
                                                   const std::vector< MOABReal >& vecS )
{
    const size_t nS = vecS.size();

    // populate the default CSR matrix first

    // first, create a triplet vector
    gko::matrix_data< MOABReal > mData( gko::dim< 2 >{ nOpRows, nOpCols } );
    auto& tripletList = mData.nonzeros;
    tripletList.reserve( nS );

    // loop over nnz and populate the sparse matrix operator
    for( size_t innz = 0; innz < nS; ++innz )
    {
        tripletList.emplace_back( vecRow[innz] - 1, vecCol[innz] - 1, vecS[innz] );
    }

    // generate the main operator
    {
        // auto mapOperator = SpMV_DefaultMatrixType::create( this->host_executor );
        // // populate the CSR sparse matrix with the matrix_data object
        // primaryMapOperator->read( mData );

        // next, let us take care of the forward operator
        mapOperator = MatrixType::create( device_executor );  // should this be "host_executor" ?
        mapOperator->read( mData );
        // if( std::is_same< SpMV_DefaultMatrixType, MatrixType >::value )
        // {
        //     // set the pointer to our original underlying CSR format matrix
        //     primaryMapOperator->move_to( mapOperator.get() );
        // }
        // else
        // {
        //     // Perform the conversion to the format requested
        //     std::cout << "> Converting matrix to " << this->matrix_type() << "\n ";
        //     primaryMapOperator->convert_to( mapOperator.get() );
        // }
    }

    if( enableTransposeOp )
    {
        /*
        mData.size = gko::dim< 2 >{ nOpCols, nOpRows };  // transpose operator sizing

        // reuse the same tripletlist reference and swap row/col indices to generate transpose operator
        // simple to use for_each algorithm with a lambda to swap
        std::for_each( std::begin( tripletList ), std::end( tripletList ),
                       []( auto& triplet ) { std::swap(triplet.row, triplet.column); } );

        // loop over nnz and populate the sparse matrix operator
        auto primaryTransposeMapOperator = SpMV_DefaultMatrixType::create( this->host_executor );
        primaryTransposeMapOperator->read( mData );

        // populate the default CSR matrix first
        mapTransposeOperator = MatrixType::create( device_executor );  // should this be "host_executor" ?
        if( !std::is_same< SpMV_DefaultMatrixType, MatrixType >::value )
        {
            // Perform the conversion to the format requested
            std::cout << "> Converting tranpose matrix to " << this->matrix_type() << "\n ";
            primaryTransposeMapOperator->convert_to( mapTransposeOperator.get() );
        }
        */

        this->compute_transpose_operator( vecRow, vecCol, vecS );
    }

    return;
}

// template < typename MatrixType >
// bool GinkgoOperator< MatrixType >::PerformVerification( const std::vector< MOABReal >& vecAreasA,
//                                                         const std::vector< MOABReal >& vecAreasB )
// {
//     if( !std::is_same< GinkgoCSRMatrix, MatrixType >::value )
//     {
//         std::cout << "Please use ginkgo:CSR as the matrix format type to perform verifications." << std::endl;
//         return false;
//     }
//     return false;
// }

template < typename MatrixType >
void GinkgoOperator< MatrixType >::apply_operator( const std::unique_ptr< SpMV_VectorType >& rhs,
                                                   std::unique_ptr< SpMV_VectorType >& result )
{
    mapOperator->apply( gko::lend( rhs ), gko::lend( result ) );
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::apply_transpose_operator( const std::unique_ptr< SpMV_VectorType >& rhs,
                                                             std::unique_ptr< SpMV_VectorType >& result )
{
    mapTransposeOperator->apply( gko::lend( rhs ), gko::lend( result ) );
}

template <>
void GinkgoOperator< GinkgoCSRMatrix >::apply_transpose_operator( const std::unique_ptr< SpMV_VectorType >& rhs,
                                                                  std::unique_ptr< SpMV_VectorType >& result )
{
    auto csrTransposeOperator = mapOperator->transpose();
    csrTransposeOperator->apply( gko::lend( rhs ), gko::lend( result ) );
}

template < typename MatrixType >
bool GinkgoOperator< MatrixType >::PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                                        const std::vector< MOABReal >& vecAreasB )
{
    assert( enableTransposeOp );
    assert( vecAreasA.size() == nOpCols );
    assert( vecAreasB.size() == nOpRows );
    bool isVerifiedAx = false, isVerifiedATx = false;

    using val_array = gko::Array< MOABReal >;

    std::cout << "\nPerforming A*x and A^T*x accuracy verifications" << std::endl;
    // Define temporary vectors to compute matrix-vector products
    {
        auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpCols, 1 } );
        auto tgtSrc = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpRows, 1 } );
        srcTgt->fill( 1.0 );
        tgtSrc->fill( 0.0 );

        // Perform the SpMV operation
        this->apply_operator( srcTgt, tgtSrc );

        // compute the error in the A*x application
        const auto tgtSrcValues = tgtSrc->get_values();
        MOABReal errorAxNrm     = 0.0;
        for( auto ind = 0; ind < nOpRows; ++ind )
        {
            const MOABReal errorAx = ( 1.0 - tgtSrcValues[ind] );
            errorAxNrm += errorAx * errorAx;
        }
        // errorAxNrm /= nOpRows;
        errorAxNrm = std::sqrt( errorAxNrm );

        isVerifiedAx = ( errorAxNrm < 1e-6 );
        std::cout << "   > A*[ones] = ones ? " << ( isVerifiedAx ? "Yes." : "No." )
                  << " Error||A*[ones] - [ones]||_2 = " << errorAxNrm << std::endl;
    }

    {
        auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpCols, 1 } );
        auto tgtSrc = SpMV_VectorType::create(
            device_executor, gko::dim< 2 >( nOpRows, 1 ),
            val_array::view( host_executor, nOpRows, const_cast< MOABReal* >( vecAreasB.data() ) ), 1 );
        srcTgt->fill( 0.0 );

        // const auto tgtSrcValues = tgtSrc->get_values();
        // std::cout << "tgtSrcValues: " << tgtSrcValues[10] << ", " << tgtSrcValues[30] << ", " << tgtSrcValues[200] <<
        // ", "
        //           << tgtSrcValues[399] << std::endl;
        // std::cout << "reference: " << vecAreasB[10] << ", " << vecAreasB[30] << ", " << vecAreasB[200] << ", "
        //           << vecAreasB[399] << std::endl;

        this->apply_transpose_operator( tgtSrc, srcTgt );

        const auto srcTgtValues = srcTgt->get_values();
        MOABReal errorATxNrm    = 0.0;
        for( auto ind = 0; ind < nOpCols; ++ind )
        {
            const MOABReal errorATx = ( vecAreasA[ind] - srcTgtValues[ind] );
            errorATxNrm += errorATx * errorATx;
        }
        // errorATxNrm /= nOpCols;
        errorATxNrm = std::sqrt( errorATxNrm );

        std::cout << "srcTgtValues: " << srcTgtValues[0] << ", " << srcTgtValues[1] << ", " << srcTgtValues[2] << ", "
                  << srcTgtValues[3] << std::endl;
        std::cout << "reference: " << vecAreasA[0] << ", " << vecAreasA[1] << ", " << vecAreasA[2] << ", "
                  << vecAreasA[3] << std::endl;

        isVerifiedATx = ( errorATxNrm < 1e-12 );
        std::cout << "   > A^T*vecAreaB = vecAreaA ? " << ( isVerifiedATx ? "Yes." : "No." )
                  << " Error||A^T*vecAreaB - vecAreaA||_2 = " << errorATxNrm << std::endl;
    }
    std::cout << std::endl;

    return ( isVerifiedAx && isVerifiedATx );
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::PerformSpMV( int n_remap_iterations )
{
    // multiple RHS for each variable to be projected
    auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpCols, nRHSV } );
    srcTgt->fill( 1.0 );
    auto tgtSrc = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpRows, nRHSV } );
    tgtSrc->fill( 0.0 );

    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        this->apply_operator( srcTgt, tgtSrc );
    }
    return;
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::PerformSpMVTranspose( int n_remap_iterations )
{
    assert( enableTransposeOp );

    // multiple RHS for each variable to be projected
    auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpCols, nRHSV } );
    srcTgt->fill( 0.0 );
    auto tgtSrc = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpRows, nRHSV } );
    tgtSrc->fill( 1.0 );

    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        this->apply_transpose_operator( tgtSrc, srcTgt );
    }
    return;
}

template <>
void GinkgoOperator< GinkgoCSRMatrix >::PerformSpMVTranspose( int n_remap_iterations )
{
    assert( enableTransposeOp );

    // multiple RHS for each variable to be projected
    auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpCols, nRHSV } );
    srcTgt->fill( 0.0 );
    auto tgtSrc = SpMV_VectorType::create( device_executor, gko::dim< 2 >{ nOpRows, nRHSV } );
    tgtSrc->fill( 1.0 );

    auto csrTransposeOperator = mapOperator->transpose();
    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        csrTransposeOperator->apply( gko::lend( tgtSrc ), gko::lend( srcTgt ) );
    }
    return;
}

template < typename MatrixType >
std::string GinkgoOperator< MatrixType >::matrix_type()
{
    if( std::is_same< GinkgoCSRMatrix, MatrixType >::value )
        return "CSR";
    else if( std::is_same< GinkgoCOOMatrix, MatrixType >::value )
        return "COO";
    else if( std::is_same< GinkgoELLMatrix, MatrixType >::value )
        return "ELL";
    else if( std::is_same< GinkgoHybridEllMatrix, MatrixType >::value )
        return "Hybrid";
    else if( std::is_same< GinkgoSellpMatrix, MatrixType >::value )
        return "SELL-P";
    else
        return "ERROR: Unknown matrix type";
}

#endif  //  __SPMVAPP_GINKGO_H__