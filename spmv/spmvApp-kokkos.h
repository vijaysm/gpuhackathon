#ifndef __SPMVAPP_KOKKOS_H__
#define __SPMVAPP_KOKKOS_H__

#ifndef USE_KOKKOS
#error "Define 'USE_KOKKOS' and ensure makefile includes header path."
#endif

// Defines for LA experiments
#include "spmvAppUtils.h"

// C++ includes
#include <algorithm>

// Kokkos includes
#include <Kokkos_Core.hpp>
#include <KokkosKernels_default_types.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_Utils.hpp>

template < typename Device >
class KokkosKernelOperator : public SpMVOperator
{
  public:
    // Define a Vector type
    typedef default_scalar Scalar;
    typedef default_lno_t Ordinal;
    typedef default_size_type Offset;
    typedef default_layout Layout;

    typedef typename Device::execution_space execution_space;
    typedef Device device_type;

    typedef Kokkos::View< const Ordinal*, device_type > row_map_type;
    typedef Kokkos::View< Scalar*, device_type > entries_type;

    typedef KokkosSparse::CrsMatrix< Scalar, Ordinal, device_type, void, Offset > SpMV_MatrixType;
    typedef KokkosSparse::CrsMatrix< Scalar, Ordinal, device_type, void, Offset > SpMV_DefaultMatrixType;
    using values_type = typename SpMV_MatrixType::values_type;

    typedef typename values_type::non_const_type SpMV_VectorType;

    // HELP: https://github.com/kokkos/kokkos-kernels/blob/master/example/wiki/sparse/KokkosSparse_wiki_crsmatrix.cpp

    SpMV_MatrixType mapOperator;
    SpMV_MatrixType mapTransposeOperator;

    KokkosKernelOperator( MOABSInt nOpRows, MOABSInt nOpCols, MOABSInt nOpNNZs, MOABSInt nVecs,
                          bool requireTransposeOp = false );
    virtual ~KokkosKernelOperator(){};  // TODO: clear operator memory
    virtual void CreateOperator(  const std::vector< MOABSInt >& vecRow,
                         const std::vector< MOABSInt >& vecCol, const std::vector< MOABReal >& vecS );
    virtual bool PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                      const std::vector< MOABReal >& vecAreasB );
    virtual void PerformSpMV( int n_remap_iterations = 1 );
    virtual void PerformSpMVTranspose( int n_remap_iterations = 1 );

    const Scalar SC_ONE = Kokkos::ArithTraits< Scalar >::one();
    const Scalar SC_ZERO = Kokkos::ArithTraits< Scalar >::zero();
};

template < typename Device >
KokkosKernelOperator< Device >::KokkosKernelOperator( MOABSInt nRows, MOABSInt nCols, MOABSInt nNNZs,
                                                                    MOABSInt nRHSV, bool requireTransposeOp )
    : SpMVOperator( nRows, nCols, nNNZs, nRHSV, requireTransposeOp )
{
}

template < typename Device >
void KokkosKernelOperator< Device >::CreateOperator( const std::vector< MOABSInt >& vecRow,
                                                                   const std::vector< MOABSInt >& vecCol,
                                                                   const std::vector< MOABReal >& vecS )
{
    // https://github.com/kokkos/kokkos-kernels/wiki/SPARSE-2%3A%3Aspmv
    using graph_type   = typename SpMV_MatrixType::staticcrsgraph_type;
    using row_map_type = typename graph_type::row_map_type;
    using entries_type = typename graph_type::entries_type;
    using values_type  = typename SpMV_MatrixType::values_type;

    // build the primary operator
    {
        // Build the row pointers and store numNNZ
        typename row_map_type::non_const_type row_map( "row pointers", nOpRows + 1 );
        typename row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view( row_map );

        for( Ordinal rowIdx = 0; rowIdx < vecRow.size(); ++rowIdx )
        {
            // Store the number of non-zeros per sparse row
            row_map_h[vecRow[rowIdx]]++;
        }
        for( Ordinal rowIdx = 1; rowIdx < nOpRows + 1; ++rowIdx )
        {
            // Store the number of non-zeros per sparse row
            row_map_h[rowIdx] += row_map_h[rowIdx - 1];
        }
        assert( vecS.size() - row_map_h[nOpRows] == 0 );
        const Offset numNNZ = vecS.size();
        Kokkos::deep_copy( row_map, row_map_h );

        typename entries_type::non_const_type entries( "column indices", numNNZ );
        typename entries_type::HostMirror entries_h = Kokkos::create_mirror_view( entries );
        typename values_type::non_const_type values( "values", numNNZ );
        typename values_type::HostMirror values_h = Kokkos::create_mirror_view( values );
        for( size_t ind = 0; ind < vecS.size(); ++ind )
        {
            entries_h( ind ) = vecCol[ind];
            values_h( ind )  = vecS[ind];
        }
        Kokkos::deep_copy( entries, entries_h );
        Kokkos::deep_copy( values, values_h );

        graph_type myGraph( entries, row_map );
        mapOperator = SpMV_MatrixType( "Primary operator", nOpRows, values, myGraph );
    }

    // store the transpose operator as well if requested
    if( enableTransposeOp )
    {
        // Build the row pointers and store numNNZ
        typename row_map_type::non_const_type row_map( "row pointers", nOpCols + 1 );
        typename row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view( row_map );

        for( Ordinal rowIdx = 0; rowIdx < vecCol.size(); ++rowIdx )
        {
            // Store the number of non-zeros per sparse row
            row_map_h[vecCol[rowIdx]]++;
        }
        for( Ordinal rowIdx = 1; rowIdx < nOpCols + 1; ++rowIdx )
        {
            // Store the number of non-zeros per sparse row
            row_map_h[rowIdx] += row_map_h[rowIdx - 1];
        }
        assert( vecS.size() - row_map_h[nOpCols] == 0 );
        const Offset numNNZ = vecS.size();
        Kokkos::deep_copy( row_map, row_map_h );

        // initialize original index locations
        std::vector< size_t > idx( vecS.size() );
        std::iota( idx.begin(), idx.end(), 0 );

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort( idx.begin(), idx.end(),
                          [&vecCol]( size_t i1, size_t i2 ) { return vecCol[i1] < vecCol[i2]; } );

        typename entries_type::non_const_type entries( "column indices", numNNZ );
        typename entries_type::HostMirror entries_h = Kokkos::create_mirror_view( entries );
        typename values_type::non_const_type values( "values", numNNZ );
        typename values_type::HostMirror values_h = Kokkos::create_mirror_view( values );
        for( size_t ind = 0; ind < vecS.size(); ++ind )
        {
            entries_h( ind ) = vecRow[idx[ind]];
            values_h( ind )  = vecS[idx[ind]];
        }
        Kokkos::deep_copy( entries, entries_h );
        Kokkos::deep_copy( values, values_h );

        graph_type myGraph( entries, row_map );
        mapTransposeOperator = SpMV_MatrixType( "Transpose operator", nOpCols, values, myGraph );
    }
    return;
}

template < typename Device >
bool KokkosKernelOperator< Device >::PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                                                        const std::vector< MOABReal >& vecAreasB )
{
    assert( vecColSum.size() == nOpCols );
    bool isVerifiedAx = false, isVerifiedATx = false;

    /*
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
        // Eigen::Map< const Eigen::VectorXd > refVector( vecAreasA.data(), vecAreasA.size() );
        // SpMV_VectorType errorATx = srcTgt - refVector;
        // isVerifiedATx            = (errorATx.norm() < 1e-12);
        // std::cout << "   > A^T*vecAreaB = vecAreaA ? " << ( isVerifiedATx ? "Yes." : "No." )
        //           << " Error||A^T*vecAreaB - vecAreaA||_2 = " << errorATx.norm() << std::endl;
    }
    std::cout << std::endl;
    */

    return ( isVerifiedAx && isVerifiedATx );
}

template < typename Device >
void KokkosKernelOperator< Device >::PerformSpMV( int n_remap_iterations )
{
    // Perform SpMV from Source to Target through operator application
    // multiply RHS for each variable to be projected
    SpMV_VectorType srcTgt( "lhs", nOpCols, nRHSV );  // srcTgt = 1.0
    for( auto iR = 0; iR < nOpCols; ++iR )
        srcTgt(iR) = 1.0;
    SpMV_VectorType tgtSrc( "rhs", nOpRows, nRHSV );  // tgtSrc = 0.0
    for( auto iR = 0; iR < nOpRows; ++iR )
        tgtSrc( iR ) = 0.0;

    const Scalar alpha = SC_ONE;
    const Scalar beta  = SC_ZERO;
    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from source to target through weight application for each variable
        KokkosSparse::spmv( "N", alpha, mapOperator, srcTgt, beta, tgtSrc );
    }
    return;
}

template < typename Device >
void KokkosKernelOperator< Device >::PerformSpMVTranspose( int n_remap_iterations )
{
    assert( enableTransposeOp );
    // Perform SpMV from Target to Source through transpose operator application
    // multiple RHS for each variable to be projected
    SpMV_VectorType srcTgt( "lhs", nOpCols, nRHSV );  // srcTgt = 0.0
    for( auto iR = 0; iR < nOpCols; ++iR )
        srcTgt( iR ) = 0.0;
    SpMV_VectorType tgtSrc( "rhs", nOpRows, nRHSV );  // tgtSrc = 1.0
    for( auto iR = 0; iR < nOpRows; ++iR )
        tgtSrc( iR ) = 1.0;

    const Scalar alpha = SC_ONE;
    const Scalar beta  = SC_ZERO;
    for( auto iR = 0; iR < n_remap_iterations; ++iR )
    {
        // Project data from target to source through transpose application for each variable
        KokkosSparse::spmv( "N", alpha, mapTransposeOperator, tgtSrc, beta, srcTgt );
    }
    return;
}

#endif // __SPMVAPP_KOKKOS_H__
