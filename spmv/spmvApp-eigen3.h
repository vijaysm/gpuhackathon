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

#include <algorithm>
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
    virtual ~Eigen3Operator(){};  // TODO: clear operator memory
    virtual void CreateOperator( const std::vector< MOABSInt >& vecRow, const std::vector< MOABSInt >& vecCol,
                                 const std::vector< MOABReal >& vecS );
    virtual bool PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                      const std::vector< MOABReal >& vecAreasB );
    virtual void PerformSpMV( const std::vector< double >& inputData, std::vector< double >& outputData );
    virtual void PerformSpMVTranspose( const std::vector< double >& inputData, std::vector< double >& outputData );
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
    std::vector< SparseEntry > tripletList;
    tripletList.reserve( nS );

    // loop over nnz and populate the sparse matrix operator
    // #pragma omp parallel for shared( tripletList )
    for( size_t innz = 0; innz < nS; ++innz )
    {
        // mapOperator.insert( vecRow[innz]-1, vecCol[innz]-1 ) = vecS[innz];
        tripletList.emplace_back( SparseEntry( vecRow[innz] - 1, vecCol[innz] - 1, vecS[innz] ) );
    }

    mapOperator.setFromTriplets( tripletList.begin(), tripletList.end() );

    mapOperator.makeCompressed();

    // store the transpose operator as well.
    if( enableTransposeOp )
    {
        mapTransposeOperator = mapOperator.transpose();

        // mapTransposeOperator.resize( nOpCols, nOpRows );
        // mapTransposeOperator.reserve( nS );
        // // simple to use for_each algorithm with a lambda to swap
        // std::for_each( std::begin( tripletList ), std::end( tripletList ),
        //                []( SparseEntry& triplet ) { std::swap( triplet.row(), triplet.col() ); } );

        // mapTransposeOperator.setFromTriplets( tripletList.begin(), tripletList.end() );
        // mapTransposeOperator.makeCompressed();
    }
    return;
}

bool Eigen3Operator::PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                          const std::vector< MOABReal >& vecAreasB )
{
    assert( vecColSum.size() == nOpCols );
    bool isVerifiedAx = false, isVerifiedATx = false;

    std::cout << "\nPerforming A*x and A^T*x accuracy verifications" << std::endl;
    // Define temporary vectors to compute matrix-vector products
    {
        SpMV_VectorType srcTgt = SpMV_VectorType::Ones( nOpCols, 1 );
        SpMV_VectorType tgtSrc = SpMV_VectorType::Zero( nOpRows, 1 );

        // Perform the SpMV operation
        tgtSrc                  = mapOperator * srcTgt;
        isVerifiedAx            = tgtSrc.isOnes( 1e-6 );
        SpMV_VectorType errorAx = tgtSrc - SpMV_VectorType::Ones( nOpRows, 1 );
        std::cout << "   > A*[ones] = ones ? " << ( isVerifiedAx ? "Yes." : "No." )
                  << " Error||A*[ones] - [ones]||_2 = " << errorAx.norm() << std::endl;
    }

    {
        SpMV_VectorType srcTgt = SpMV_VectorType::Zero( nOpCols, 1 );
        // SpMV_VectorType tgtSrc = SpMV_VectorType::Ones( nOpRows, 1 );
        Eigen::Map< const Eigen::VectorXd > tgtSrc( vecAreasB.data(), vecAreasB.size() );

        // const auto tgtSrcValues = tgtSrc->get_values();
        // std::cout << "tgtSrcValues: " << tgtSrc( 10 ) << ", " << tgtSrc( 30 ) << ", " << tgtSrc( 200 ) << ", "
        //           << tgtSrc( 399 ) << std::endl;
        // std::cout << "reference: " << vecAreasB[10] << ", " << vecAreasB[30] << ", " << vecAreasB[200] << ", "
        //           << vecAreasB[399] << std::endl;

        // Perform the tranpose SpMV operation
        if( enableTransposeOp ) { srcTgt = mapTransposeOperator * tgtSrc; }
        else
        {
            SpMV_MatrixType transposeMap = mapOperator.transpose();
            srcTgt                       = transposeMap * tgtSrc;
        }

        // std::cout << "srcTgtValues: " << srcTgt(0) << ", " << srcTgt(1) << ", " << srcTgt(2) << ", " << srcTgt(3)
        //           << std::endl;
        // std::cout << "reference: " << vecAreasA[0] << ", " << vecAreasA[1] << ", " << vecAreasA[2] << ", "
        //           << vecAreasA[3] << std::endl;
        Eigen::Map< const Eigen::VectorXd > refVector( vecAreasA.data(), vecAreasA.size() );
        SpMV_VectorType errorATx = srcTgt - refVector;
        isVerifiedATx            = ( errorATx.norm() < 1e-12 );
        std::cout << "   > A^T*vecAreaB = vecAreaA ? " << ( isVerifiedATx ? "Yes." : "No." )
                  << " Error||A^T*vecAreaB - vecAreaA||_2 = " << errorATx.norm() << std::endl;
    }
    std::cout << std::endl;

    return ( isVerifiedAx && isVerifiedATx );
}

void Eigen3Operator::PerformSpMV( const std::vector< double >& inputData, std::vector< double >& outputData )
{
    // Perform SpMV from Source to Target through operator application
    // multiply RHS for each variable to be projected
    Eigen::Map< const SpMV_VectorType > xRhs( inputData.data(), nOpCols, nRHSV );
    Eigen::Map< SpMV_VectorType > xOut( outputData.data(), nOpRows, nRHSV );

    // Project data from source to target through weight application for each variable
    {
        xOut = mapOperator * xRhs;
        // #pragma omp parallel for
        //         for( auto iVar = 0; iVar < nRHSV; ++iVar )
        //             tgtSrc.col( iVar ) = mapOperator * srcTgt.col( iVar );
    }
    return;
}

void Eigen3Operator::PerformSpMVTranspose( const std::vector< double >& inputData, std::vector< double >& outputData )
{
    assert( enableTransposeOp );
    // Perform SpMV from Target to Source through transpose operator application
    // multiple RHS for each variable to be projected
    Eigen::Map< const SpMV_VectorType > xRhs( inputData.data(), nOpRows, nRHSV );
    Eigen::Map< SpMV_VectorType > xOut( outputData.data(), nOpCols, nRHSV );

    // Project data from target to source through transpose application for each variable
    {
        xOut = mapTransposeOperator * xRhs;
        // #pragma omp parallel for
        //         for( auto iVar = 0; iVar < nRHSV; ++iVar )
        //             srcTgt.col( iVar ) = mapTransposeOperator * tgtSrc.col( iVar );
    }
    return;
}

#endif
