#ifndef __SPMVAPP_KOKKOS_H__
#define __SPMVAPP_KOKKOS_H__

#ifndef USE_KOKKOS
#error "Define 'USE_KOKKOS' and ensure makefile includes header path."
#endif

// Defines for LA experiments
#include "spmvAppUtils.h"

// C++ includes
#include <algorithm>
#include <numeric>

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

    // typedef typename values_type::non_const_type SpMV_VectorType;
    typedef Kokkos::View< Scalar**, execution_space > SpMV_VectorType;

    // HELP: https://github.com/kokkos/kokkos-kernels/blob/master/example/wiki/sparse/KokkosSparse_wiki_crsmatrix.cpp

    SpMV_MatrixType mapOperator;
    SpMV_MatrixType mapTransposeOperator;

    SpMV_VectorType forwardRhs;
    SpMV_VectorType reverseRhs;
    typename SpMV_VectorType::HostMirror h_reverseRhs;
    typename SpMV_VectorType::HostMirror h_forwardRhs;

    KokkosKernelOperator( MOABSInt nOpRows, MOABSInt nOpCols, MOABSInt nOpNNZs, MOABSInt nVecs,
                          bool requireTransposeOp = false );
    virtual ~KokkosKernelOperator(){};  // TODO: clear operator memory
    virtual void CreateOperator( const std::vector< MOABSInt >& vecRow, const std::vector< MOABSInt >& vecCol,
                                 const std::vector< MOABReal >& vecS );
    virtual bool PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                      const std::vector< MOABReal >& vecAreasB );
    virtual void PerformSpMV( const std::vector< double >& inputData, std::vector< double >& outputData );
    virtual void PerformSpMVTranspose( const std::vector< double >& inputData, std::vector< double >& outputData );

  private:
    void apply_operator( const SpMV_VectorType& rhs, SpMV_VectorType& result );
    void apply_transpose_operator( const SpMV_VectorType& rhs, SpMV_VectorType& result );

    const Scalar SC_ONE  = Kokkos::ArithTraits< Scalar >::one();
    const Scalar SC_ZERO = Kokkos::ArithTraits< Scalar >::zero();
};

template < typename Device >
KokkosKernelOperator< Device >::KokkosKernelOperator( MOABSInt nRows, MOABSInt nCols, MOABSInt nNNZs, MOABSInt nRHSV,
                                                      bool requireTransposeOp )
    : SpMVOperator( nRows, nCols, nNNZs, nRHSV, requireTransposeOp )
{
    std::cout << "device_type: " << execution_space::name() << std::endl;
    forwardRhs = SpMV_VectorType( "lhs", nCols, nRHSV );
    reverseRhs = SpMV_VectorType( "rhs", nRows, nRHSV );

    h_reverseRhs = Kokkos::create_mirror_view( reverseRhs );
    h_forwardRhs = Kokkos::create_mirror_view( forwardRhs );
    // std::cout << forwardRhs.data() << " - " << forwardRhs.size() << std::endl;
    // std::cout << reverseRhs.data() << " - " << reverseRhs.size() << std::endl;
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

    size_t nNNZ = vecS.size();

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
            entries_h( ind ) = vecCol[ind] - 1;
            values_h( ind )  = vecS[ind];
        }
        Kokkos::deep_copy( entries, entries_h );
        Kokkos::deep_copy( values, values_h );

        // graph_type myGraph( entries, row_map );
        // mapOperator = SpMV_MatrixType( "Primary operator", nOpRows, values, myGraph );
        mapOperator = SpMV_MatrixType( "Primary operator", nOpRows, nOpCols, nNNZ, values, row_map, entries );
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
            entries_h( ind ) = vecRow[idx[ind]] - 1;
            values_h( ind )  = vecS[idx[ind]];
        }
        Kokkos::deep_copy( entries, entries_h );
        Kokkos::deep_copy( values, values_h );

        // graph_type myGraph( entries, row_map );
        // mapTransposeOperator = SpMV_MatrixType( "Transpose operator", nOpCols, nOpRows, values, myGraph );
        mapTransposeOperator =
            SpMV_MatrixType( "Transpose operator", nOpCols, nOpRows, nNNZ, values, row_map, entries );
    }
    return;
}

template < typename Device >
bool KokkosKernelOperator< Device >::PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                                          const std::vector< MOABReal >& vecAreasB )
{
    if( !enableTransposeOp )
    {
        std::cout << "Enable transpose operation to verify both A*x and A^T*x correctly.\n";
        return false;
    }

    assert( vecColSum.size() == nOpCols );
    bool isVerifiedAx = false, isVerifiedATx = false;

    std::cout << "\nPerforming A*x and A^T*x accuracy verifications" << std::endl;
    // Define temporary vectors to compute matrix-vector products
    {
        SpMV_VectorType srcTgt( "lhs", nOpCols, 1 );  // srcTgt = 1.0
        for( int iR = 0; iR < nOpCols; ++iR )
            srcTgt( iR, 0 ) = 1.0;
        SpMV_VectorType tgtSrc( "rhs", nOpRows, 1 );  // tgtSrc = 0.0
        for( int iR = 0; iR < nOpRows; ++iR )
            tgtSrc( iR, 0 ) = 0.0;

        // Perform the SpMV operation
        this->apply_operator( srcTgt, tgtSrc );
        Scalar errorAx = 0.0;
        for( int iR = 0; iR < nOpRows; ++iR )
            errorAx += std::pow( tgtSrc( iR, 0 ) - 1.0, 2.0 );
        errorAx      = std::sqrt( errorAx );
        isVerifiedAx = ( errorAx < 1e-6 );
        std::cout << "   > A*[ones] = ones ? " << ( isVerifiedAx ? "Yes." : "No." )
                  << " Error||A*[ones] - [ones]||_2 = " << errorAx << std::endl;
    }

    {
        SpMV_VectorType srcTgt( "lhs", nOpCols, 1 );  // srcTgt = 0.0
        for( int iR = 0; iR < nOpCols; ++iR )
            srcTgt( iR, 0 ) = 0.0;
        SpMV_VectorType tgtSrc( "rhs", nOpRows, 1 );  // tgtSrc = vecAreasB = target areas
        for( int iR = 0; iR < nOpRows; ++iR )
            tgtSrc( iR, 0 ) = vecAreasB[iR];

        // const auto tgtSrcValues = tgtSrc->get_values();
        // std::cout << "tgtSrcValues: " << tgtSrc( 10 ) << ", " << tgtSrc( 30 ) << ", " << tgtSrc( 200 ) << ", "
        //           << tgtSrc( 399 ) << std::endl;
        // std::cout << "reference: " << vecAreasB[10] << ", " << vecAreasB[30] << ", " << vecAreasB[200] << ", "
        //           << vecAreasB[399] << std::endl;

        // Perform the tranpose SpMV operation
        this->apply_transpose_operator( tgtSrc, srcTgt );

        std::cout << "srcTgtValues: " << srcTgt( 0, 0 ) << ", " << srcTgt( 1, 0 ) << ", " << srcTgt( 2, 0 ) << ", "
                  << srcTgt( 3, 0 ) << std::endl;
        std::cout << "reference: " << vecAreasA[0] << ", " << vecAreasA[1] << ", " << vecAreasA[2] << ", "
                  << vecAreasA[3] << std::endl;

        Scalar errorATx = 0.0;
        for( int iR = 0; iR < nOpCols; ++iR )
            errorATx += std::pow( srcTgt( iR, 0 ) - vecAreasA[iR], 2.0 );  // now srcTgt = reference vector
        errorATx = std::sqrt( errorATx );

        isVerifiedATx = ( errorATx < 1e-12 );
        std::cout << "   > A^T*vecAreaB = vecAreaA ? " << ( isVerifiedATx ? "Yes." : "No." )
                  << " Error||A^T*vecAreaB - vecAreaA||_2 = " << errorATx << std::endl;
    }
    std::cout << std::endl;

    return ( isVerifiedAx && isVerifiedATx );
}

template < typename Device >
void KokkosKernelOperator< Device >::apply_operator( const SpMV_VectorType& rhs, SpMV_VectorType& result )
{
    const Scalar alpha = SC_ONE;
    const Scalar beta  = SC_ZERO;

    KokkosSparse::spmv( "N", alpha, mapOperator, rhs, beta, result );
}

template < typename Device >
void KokkosKernelOperator< Device >::apply_transpose_operator( const SpMV_VectorType& rhs, SpMV_VectorType& result )
{
    const Scalar alpha = SC_ONE;
    const Scalar beta  = SC_ZERO;

    KokkosSparse::spmv( "N", alpha, mapTransposeOperator, rhs, beta, result );
}

template < typename Device >
void KokkosKernelOperator< Device >::PerformSpMV( const std::vector< double >& inputData,
                                                  std::vector< double >& outputData )
{
    assert( this->nRHSV * this->nOpCols == inputdata.size() );

    int l_nRHSV   = this->nRHSV;
    int l_nOpCols = this->nOpCols;
    int l_nOpRows = this->nOpRows;

    // Kokkos::View<double**, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > inputView ( "SpMVInputView",
    // inputdata.data(), this->nOpCols, this->nRHSV);

    auto l_fwdRhs          = forwardRhs;
    using MDRangePolicy_2D = Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank< 2 > >;
    MDRangePolicy_2D policy( { 0, 0 }, { l_nOpCols, l_nRHSV } );
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA( const int iR, const int iV ) { l_fwdRhs( iR, iV ) = 1.0; } );

    Kokkos::deep_copy( reverseRhs, 0.0 );

    // Project data from source to target through weight application for each variable
    {
        // KokkosSparse::spmv( "N", alpha, mapOperator, srcTgt, beta, tgtSrc );
        this->apply_operator( forwardRhs, reverseRhs );

        // Copy the result back to data vector
        Kokkos::deep_copy( h_reverseRhs, reverseRhs );
    }
    return;
}

Kokkos::deep_copy( tgtSrc, 0.0 );

template < typename Device >
void KokkosKernelOperator< Device >::PerformSpMVTranspose( const std::vector< double >& inputData,
                                                           std::vector< double >& outputData )
{
    assert( enableTransposeOp );
    int l_nRHSV   = this->nRHSV;
    int l_nOpCols = this->nOpCols;
    int l_nOpRows = this->nOpRows;

    auto l_fwdRhs          = forwardRhs;
    using MDRangePolicy_2D = Kokkos::Experimental::MDRangePolicy< Kokkos::Experimental::Rank< 2 > >;
    /*
    MDRangePolicy_2D policy({0,0}, {l_nOpCols, l_nRHSV});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int iR, const int iV)
    {
        l_fwdRhs(iR, iV) = 0.0;
    });
    */
    Kokkos::deep_copy( forwardRhs, 0.0 );

    auto l_revRhs = reverseRhs;
    MDRangePolicy_2D policy2( { 0, 0 }, { l_nOpRows, l_nRHSV } );
    Kokkos::parallel_for(
        policy2, KOKKOS_LAMBDA( const int iR, const int iV ) { l_revRhs( iR, iV ) = 1.0; } );

    {
        // Project data from target to source through transpose application for each variable
        // KokkosSparse::spmv( "N", alpha, mapTransposeOperator, tgtSrc, beta, srcTgt );
        this->apply_transpose_operator( reverseRhs, forwardRhs );

        Kokkos::deep_copy( h_forwardRhs, forwardRhs );
    }
    return;
}

#endif  // __SPMVAPP_KOKKOS_H__
