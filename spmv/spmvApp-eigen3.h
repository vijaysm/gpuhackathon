#ifndef __SPMVAPP_EIGEN3_H__
#define __SPMVAPP_EIGEN3_H__

#ifndef USE_EIGEN3
#error "Define 'USE_EIGEN3' and ensure makefile includes header path."
#endif

// Defines for LA experiments
#include "spmvAppUtils.h"

// workaround issue between gcc >= 4.7 and cuda 5.5
// https://eigen.tuxfamily.org/dox/TopicCUDA.html
#if( defined __GNUC__ ) && ( __GNUC__ > 4 || __GNUC_MINOR__ >= 7 )
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

// Do we need the below ?
// #define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include <Eigen/Dense>
#include <Eigen/Sparse>

class Eigen3Operator : public SpMVOperator
{
  public:
    // Define a Vector type
    typedef Eigen::MatrixXd SpMV_VectorType;
    typedef Eigen::SparseMatrix< MOABReal > SpMV_MatrixType;
    typedef Eigen::SparseMatrix< MOABReal > SpMV_DefaultMatrixType;

    SpMV_MatrixType mapOperator;
    SpMV_MatrixType mapTransposeOperator;

    Eigen3Operator( MOABSInt nOpRows, MOABSInt nOpCols, MOABSInt nOpNNZs, MOABSInt nVecs,
                    bool requireTransposeOp = false );
    virtual ~Eigen3Operator() {}; // TODO: clear operator memory
    virtual void CreateOperator(  const std::vector< MOABSInt >& vecRow,
                         const std::vector< MOABSInt >& vecCol, const std::vector< MOABReal >& vecS );
    virtual void PerformSpMV( int n_remap_iterations = 1 );
    virtual void PerformSpMVTranspose( int n_remap_iterations = 1 );
};

Eigen3Operator::Eigen3Operator( MOABSInt nRows, MOABSInt nCols, MOABSInt nNNZs, MOABSInt nRHSV,
                                bool requireTransposeOp )
    : SpMVOperator( nRows, nCols, nNNZs, nRHSV, requireTransposeOp )
{
}

void Eigen3Operator::CreateOperator( const std::vector< MOABSInt >& vecRow, const std::vector< MOABSInt >& vecCol,
                                     const std::vector< MOABReal >& vecS )
{
    const size_t nS = vecS.size();
    // Let us populate the map object for every process
    mapOperator.resize( nOpRows, nOpCols );
    mapOperator.reserve( nS );

    // create a triplet vector
    typedef Eigen::Triplet< MOABReal > SparseEntry;
    std::vector< SparseEntry > tripletList( nS );

    // loop over nnz and populate the sparse matrix operator
    for( size_t innz = 0; innz < nS; ++innz )
    {
        // mapOperator.insert( vecRow[innz]-1, vecCol[innz]-1 ) = vecS[innz];
        tripletList[innz] = SparseEntry( vecRow[innz] - 1, vecCol[innz] - 1, vecS[innz] );
    }

    mapOperator.setFromTriplets( tripletList.begin(), tripletList.end() );

    mapOperator.makeCompressed();

    // store the transpose operator as well.
    if( enableTransposeOp ) mapTransposeOperator = mapOperator.transpose();
    return;
}

void Eigen3Operator::PerformSpMV( int n_remap_iterations )
{
    // Perform SpMV from Source to Target through operator application
    // multiply RHS for each variable to be projected
    SpMV_VectorType srcTgt = SpMV_VectorType::Random( nOpCols, nRHSV );
    SpMV_VectorType tgtSrc = SpMV_VectorType::Zero( nOpRows, nRHSV );

    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        for( auto iVar = 0; iVar < nRHSV; ++iVar )
            tgtSrc.col( iVar ) = mapOperator * srcTgt.col( iVar );
    }
    return;
}

void Eigen3Operator::PerformSpMVTranspose( int n_remap_iterations )
{
    assert( enableTransposeOp );
    // Perform SpMV from Target to Source through transpose operator application
    // multiple RHS for each variable to be projected
    SpMV_VectorType srcTgt = SpMV_VectorType::Zero( nOpCols, nRHSV );
    SpMV_VectorType tgtSrc = SpMV_VectorType::Random( nOpRows, nRHSV );

    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from target to source through transpose application for each variable
        for( auto iVar = 0; iVar < nRHSV; ++iVar )
            srcTgt.col( iVar ) = mapTransposeOperator * tgtSrc.col( iVar );
    }
    return;
}

#endif
