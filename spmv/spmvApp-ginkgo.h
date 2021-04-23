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
void GinkgoOperator< MatrixType >::CreateOperator( const std::vector< MOABSInt >& vecRow,
                                                   const std::vector< MOABSInt >& vecCol,
                                                   const std::vector< MOABReal >& vecS )
{
    const size_t nS = vecS.size();

    // populate the default CSR matrix first

    // first, create a triplet vector
    gko::matrix_data< MOABReal > mData( gko::dim< 2 >( nOpRows, nOpCols ) );
    auto& tripletList = mData.nonzeros;
    tripletList.reserve( nS );

    // loop over nnz and populate the sparse matrix operator
    for( size_t innz = 0; innz < nS; ++innz )
    {
        tripletList.emplace_back( vecRow[innz] - 1, vecCol[innz] - 1, vecS[innz] );
    }

    // generate the main operator
    {
        auto primaryMapOperator = SpMV_DefaultMatrixType::create( this->host_executor );
        // populate the CSR sparse matrix with the matrix_data object
        primaryMapOperator->read( mData );

        // next, let us take care of the forward operator
        mapOperator = MatrixType::create( device_executor );  // should this be "host_executor" ?
        if( std::is_same< SpMV_DefaultMatrixType, MatrixType >::value )
        {
            // set the pointer to our original underlying CSR format matrix
            primaryMapOperator->move_to( mapOperator.get() );
        }
        else
        {
            // Perform the conversion to the format requested
            std::cout << "> Converting matrix to " << this->matrix_type() << "\n ";
            primaryMapOperator->convert_to( mapOperator.get() );
        }
    }

    if( enableTransposeOp )
    {
        mData.size = gko::dim< 2 >( nOpCols, nOpRows );  // transpose operator sizing

        // reuse the same tripletlist reference and swap row/col indices to generate transpose operator
        // simple to use for_each algorithm with a lambda to swap
        std::for_each( std::begin( tripletList ), std::end( tripletList ),
                       []( auto& triplet ) { std::swap(triplet.row, triplet.column); } );

        // loop over nnz and populate the sparse matrix operator
        auto primaryTransposeMapOperator = SpMV_DefaultMatrixType::create( this->host_executor );
        primaryTransposeMapOperator->read( mData );

        // populate the default CSR matrix first
        mapTransposeOperator = MatrixType::create( device_executor );  // should this be "host_executor" ?
        if( std::is_same< SpMV_DefaultMatrixType, MatrixType >::value )
        {
            // set the pointer to our original underlying CSR format matrix
            primaryTransposeMapOperator->move_to( mapTransposeOperator.get() );
        }
        else
        {
            // Perform the conversion to the format requested
            std::cout << "> Converting tranpose matrix to " << this->matrix_type() << "\n ";
            primaryTransposeMapOperator->convert_to( mapTransposeOperator.get() );
        }
    }

    return;
}

template < typename MatrixType >
bool GinkgoOperator< MatrixType >::PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                                        const std::vector< MOABReal >& vecAreasB )
{
    assert( vecColSum.size() == nOpCols );
    return true;
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::PerformSpMV( int n_remap_iterations )
{
    // multiple RHS for each variable to be projected
    auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >( nOpCols, nRHSV ) );
    srcTgt->fill( 1.0 );
    auto tgtSrc = SpMV_VectorType::create( device_executor, gko::dim< 2 >( nOpRows, nRHSV ) );
    tgtSrc->fill( 0.0 );

    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        mapOperator->apply( gko::lend( srcTgt ), gko::lend( tgtSrc ) );
    }
    return;
}

template < typename MatrixType >
void GinkgoOperator< MatrixType >::PerformSpMVTranspose( int n_remap_iterations )
{
    assert( enableTransposeOp );

    // multiple RHS for each variable to be projected
    auto srcTgt = SpMV_VectorType::create( device_executor, gko::dim< 2 >( nOpCols, nRHSV ) );
    srcTgt->fill( 0.0 );
    auto tgtSrc = SpMV_VectorType::create( device_executor, gko::dim< 2 >( nOpRows, nRHSV ) );
    tgtSrc->fill( 1.0 );

    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        mapTransposeOperator->apply( gko::lend( tgtSrc ), gko::lend( srcTgt ) );
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
    else return "ERROR: Unknown matrix type";
}

#endif  //  __SPMVAPP_GINKGO_H__