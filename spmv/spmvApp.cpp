/** @example spmvApp.cpp
 * \brief This example reads two map files to project a field in forward and reverse
 * direction by computing projections via sparse Matrix-Vector products between source
 * component and target component. This is strictly an example to work on single
 * node and we do not care about MPI parallelism in this experiment.
 *
 * Usage:
 *      ./spmvApp srcRemapWeightFile -v 1 -t -n iterations
 *
 *          -v represents the RHS vector size
 *          -t specifies that both A*x and A^T*x operations need to be profiled
 *          -n number of times to repeat the SpMV operation to get average time
 *
 * Note: Some datasets for forward and reverse maps have been uploaded to:
 *       https://ftp.mcs.anl.gov/pub/fathom/MeshFiles/maps/
 */

#include "moab/MOABConfig.h"
#include "moab/ErrorHandler.hpp"
#include "moab/ProgOptions.hpp"

#ifndef MOAB_HAVE_NETCDF
#error Require NetCDF to read the map files
#else
#include "netcdfcpp.h"
#endif

// Local includes
#include "spmvAppUtils.h"

// Defines for LA experiments
#define USE_EIGEN3
// #define USE_GINKGO
// #define USE_KOKKOS

#ifdef USE_EIGEN3
#include "spmvApp-eigen3.h"
#endif

#ifdef USE_GINKGO
#include "spmvApp-ginkgo.h"
#endif

// C++ includes
#include <iostream>


moab::ErrorCode ReadRemapOperator( const std::string& strMapFile, std::vector< MOABSInt >& vecRow,
                                   std::vector< MOABSInt >& vecCol, std::vector< MOABReal >& vecS, MOABSInt& nRows,
                                   MOABSInt& nCols, MOABSInt& nNZs );

void print_metrics( MOABSInt nRows, MOABSInt nCols, const std::vector< MOABSInt >& vecRow,
                    const std::vector< MOABSInt >& vecCol, const std::vector< MOABReal >& vecS );

std::unique_ptr< SpMVOperator > BuildOperatorObject( std::string operatorFamily, MOABSInt nOpRows, MOABSInt nOpCols,
                                                     MOABSInt nOpNNZs, MOABSInt nVecs, bool enableTransposeOp );

int main( int argc, char** argv )
{
    std::string remap_operator_filename = "";
    bool is_target_transposed           = false;
    MOABSInt n_remap_iterations         = 100;
    MOABSInt rhsvsize                   = 1;
    std::string operator_type           = "eigen3";

    ProgOptions opts( "Remap SpMV Mini-App" );
    opts.addRequiredArg< std::string >( "MapOperator", "Remap weights filename to optimize for SpMV",
                                        &remap_operator_filename );
    opts.addOpt< std::string >(
        "package,p",
        "Matrix package type specification for SpMV (default=\"eigen3\"). Possible values: [\"eigen3\", \"kokkos\", "
        "\"ginkgo:CSR\", \"ginkgo:COO\", \"ginkgo:ELL\", \"ginkgo:HYB\", \"ginkgo:SEP\"]",
        &operator_type );

    // opts.addOpt< std::string >( "srcmap,s", "Source map file name for projection", &remap_operator_filename );
    opts.addOpt< void >( "transpose,t", "Compute the tranpose operator application as well", &is_target_transposed );

    // Need option handling here for input filename
    opts.addOpt< MOABSInt >( "vsize,v", "Number of vectors to project (default=1)", &rhsvsize );
    opts.addOpt< MOABSInt >( "iterations,n",
                             "Number of iterations to perform to get the average performance profile (default=100)",
                             &n_remap_iterations );

    opts.parseCommandLine( argc, argv );

    // Print problem parameter details
    std::cout << "    SpMV-Remap Application" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    std::cout << "Source map             = " << remap_operator_filename << std::endl;
    std::cout << "Compute transpose map  = " << ( is_target_transposed ? "Yes" : "No" ) << std::endl;
    std::cout << "Number of iterations   = " << n_remap_iterations << std::endl;
    std::cout << std::endl;

    HighresTimer timer;
    moab::ErrorCode rval;
    MOABSInt nOpRows, nOpCols, nOpNNZs;
    std::unique_ptr<SpMVOperator> opImpl;

    // compute source data
    {
        // COO : Coordinate SparseMatrix format
        std::vector< MOABSInt > opNNZRows, opNNZCols;
        std::vector< MOABReal > opNNZVals;
        timer.push();
        rval = ReadRemapOperator( remap_operator_filename, opNNZRows, opNNZCols, opNNZVals, nOpRows, nOpCols, nOpNNZs );MB_CHK_ERR( rval );
        timer.pop( "ReadRemapOperator", true );

        print_metrics( nOpRows, nOpCols, opNNZRows, opNNZCols, opNNZVals );

        // Create the SpMV operator by initializing with appropriate sizes for allocation
        opImpl = BuildOperatorObject( operator_type, nOpRows, nOpCols, nOpNNZs, rhsvsize, is_target_transposed );

        if( opImpl )
        {
            timer.push();
            // Build the SpMV operator by setting the SparseMatrix with the triplets
            opImpl->CreateOperator( opNNZRows, opNNZCols, opNNZVals );
            timer.pop( "SetRemapOperator", true );
        }
    }

    // Perform the matrix vector products and time it accurately
    if( opImpl )
    {
        timer.push();
        for( auto iR = 0; iR < n_remap_iterations; ++iR )
        {
            opImpl->PerformSpMV();
        }
        timer.pop( "RemapTotalSpMV" );

        const MOABReal totalCPU_MS = static_cast< MOABReal >( timer.elapsed( "RemapTotalSpMV" ) ) / ( 1E6 );
        std::cout << "Average time (milli-secs) taken for " << n_remap_iterations
                  << " RemapOperator: SpMV(1) = " << totalCPU_MS / ( n_remap_iterations * rhsvsize ) << " and SpMV("
                  << rhsvsize << ") = " << totalCPU_MS / ( n_remap_iterations ) << std::endl;

        if( is_target_transposed )
        {
            timer.push();
            for( auto iR = 0; iR < n_remap_iterations; ++iR )
            {
                opImpl->PerformSpMVTranspose();
            }
            timer.pop( "RemapTransposeTotalSpMV" );

            const MOABReal totalTCPU_MS =
                static_cast< MOABReal >( timer.elapsed( "RemapTransposeTotalSpMV" ) ) / ( 1E6 );
            std::cout << "Average time (milli-secs) taken for " << n_remap_iterations
                      << " RemapOperator: SpMV-Transpose(1) = " << totalTCPU_MS / ( n_remap_iterations * rhsvsize )
                      << " and SpMV-Transpose(" << rhsvsize << ") = " << totalTCPU_MS / ( n_remap_iterations )
                      << std::endl;
        }
    }
    else
    {
        std::cout << "Nothing to do. Please enable Eigen3, Kokkos-Kernels or Ginkgo to perform the benchmark tests.\n";
    }

    return 0;
}

std::unique_ptr< SpMVOperator > BuildOperatorObject( std::string operatorFamily, MOABSInt nOpRows, MOABSInt nOpCols,
                                                     MOABSInt nOpNNZs, MOABSInt nVecs, bool enableTransposeOp )
{
    // Check the \p operatorFamily requested
    //
    // Possible options:
    //      1. "eigen3"     - CSR format
    //      2. "kokkos"     - CSR format
    //      3. "ginkgo:CSR" - CSR format
    //      4. "ginkgo:COO" - COO format
    //      5. "ginkgo:ELL" - ELL format
    //      6. "ginkgo:HYB" - Hybrid format
    //      7. "ginkgo:SEP" - Sell-P format
    assert( operatorFamily.size() == 6 || operatorFamily.size() == 10 );
    const std::string package = operatorFamily.substr(0, 6);
    const std::string matrixtype = (operatorFamily.size() == 10 ? operatorFamily.substr( 7, 3 ) : "");

    if( !package.compare( "eigen3" ) )
    {
#ifdef USE_EIGEN3
        std::cout << "> Building the Eigen3 *CSR* operator\n";
        return std::unique_ptr< SpMVOperator >(
            new Eigen3Operator( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
#else
        std::cout << "Error: Requested operatorFamily = *" << operatorFamily << "* and package = *" << package
                  << "* and matrixtype = *" << matrixtype << "*. Returning null operator\n";
#endif
    }
#ifdef USE_KOKKOS
    else if( !package.compare( "kokkos" ) )
    {
        std::cout << "> Building the Kokkos-Kernels *CSR* operator\n";
        return std::unique_ptr< SpMVOperator >(
            new KokkosKernelsOperator( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
    }
#endif
    else
    {
#ifdef USE_GINKGO
        assert( !package.compare( "ginkgo" ) );
        assert( matrixtype.size() );
        std::cout << "> Building the Ginkgo *" << matrixtype << "* operator\n";

        if( !matrixtype.compare( "CSR" ) )
            return std::unique_ptr< SpMVOperator >(
                new GinkgoOperator< GinkgoCSRMatrix >( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
        else if( !matrixtype.compare( "COO" ) )
            return std::unique_ptr< SpMVOperator >(
                new GinkgoOperator< GinkgoCOOMatrix >( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
        else if( !matrixtype.compare( "ELL" ) )
            return std::unique_ptr< SpMVOperator >(
                new GinkgoOperator< GinkgoCOOMatrix >( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
        else if( !matrixtype.compare( "HYB" ) )
            return std::unique_ptr< SpMVOperator >(
                new GinkgoOperator< GinkgoCOOMatrix >( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
        else if( !matrixtype.compare( "SEP" ) )
            return std::unique_ptr< SpMVOperator >(
                new GinkgoOperator< GinkgoCOOMatrix >( nOpRows, nOpCols, nOpNNZs, nVecs, enableTransposeOp ) );
#else
        std::cout << "Error: Requested operatorFamily = *" << operatorFamily << "* and package = *" << package
                  << "* and matrixtype = *" << matrixtype << "*. Returning null operator\n";
#endif
    }
    return 0;
}

moab::ErrorCode ReadRemapOperator( const std::string& strMapFile, std::vector< MOABSInt >& vecRow,
                                   std::vector< MOABSInt >& vecCol, std::vector< MOABReal >& vecS, MOABSInt& nRows,
                                   MOABSInt& nCols, MOABSInt& nNZs )
{
    NcError error( NcError::silent_nonfatal );
    using namespace moab;

    NcVar *varRow = NULL, *varCol = NULL, *varS = NULL;
    int nS = 0, nA = 0, nB = 0;

    // Create the NetCDF C++ interface
    NcFile ncMap( strMapFile.c_str(), NcFile::ReadOnly );

    // Read SparseMatrix entries
    NcDim* dimNA = ncMap.get_dim( "n_a" );
    if( dimNA == NULL )
    {
        MB_CHK_SET_ERR( MB_FAILURE, "Map file " << strMapFile << " does not contain dimension 'nA'" );
    }

    NcDim* dimNB = ncMap.get_dim( "n_b" );
    if( dimNB == NULL )
    {
        MB_CHK_SET_ERR( MB_FAILURE, "Map file " << strMapFile << " does not contain dimension 'nB'" );
    }

    NcDim* dimNS = ncMap.get_dim( "n_s" );
    if( dimNS == NULL )
    {
        MB_CHK_SET_ERR( MB_FAILURE, "Map file " << strMapFile << " does not contain dimension 'nS'" );
    }

    // store total number of nonzeros
    nS = dimNS->size();
    nA = dimNA->size();
    nB = dimNB->size();

    varRow = ncMap.get_var( "row" );
    if( varRow == NULL )
    {
        MB_CHK_SET_ERR( MB_FAILURE, "Map file " << strMapFile << " does not contain variable 'row'" );
    }

    varCol = ncMap.get_var( "col" );
    if( varCol == NULL )
    {
        MB_CHK_SET_ERR( MB_FAILURE, "Map file " << strMapFile << " does not contain variable 'col'" );
    }

    varS = ncMap.get_var( "S" );
    if( varS == NULL ) { MB_CHK_SET_ERR( MB_FAILURE, "Map file " << strMapFile << " does not contain variable 'S'" ); }

    // Resize the vectors to hold row/col indices and nnz values
    const long offsetRead = 0;
    vecRow.resize( nS );
    vecCol.resize( nS );
    vecS.resize( nS );

    varRow->set_cur( offsetRead );
    varRow->get( &( vecRow[0] ), nS );

    varCol->set_cur( offsetRead );
    varCol->get( &( vecCol[0] ), nS );

    varS->set_cur( offsetRead );
    varS->get( &( vecS[0] ), nS );

    ncMap.close();

    nRows = nB;
    nCols = nA;
    nNZs  = nS;

    return moab::MB_SUCCESS;
}

void print_metrics( MOABSInt nRows, MOABSInt nCols, const std::vector< MOABSInt >& vecRow,
                    const std::vector< MOABSInt >& /* vecCol */, const std::vector< MOABReal >& /* vecS */ )
{
    const int MAXNNZ = 500;
    std::cout << "Analyzing remap operator: size = [" << nRows << " x " << nCols << "] with NNZ = " << vecRow.size()
              << "\n";
    std::vector< int > nnzPerRow( nRows, 0 );
    std::vector< int > nnzPerRowHist( MAXNNZ, -1 );
    int maxNNZperRow = 0;
    int minNNZperRow = MAXNNZ;

    for( size_t iR = 0; iR < vecRow.size(); ++iR )
    {
        nnzPerRow[vecRow[iR] - 1]++;
    }
    for( size_t iR = 0; iR < nnzPerRow.size(); ++iR )
    {
        // Compute maxima
        maxNNZperRow = ( maxNNZperRow > nnzPerRow[iR] ? maxNNZperRow : nnzPerRow[iR] );
        // Compute minima
        minNNZperRow = ( minNNZperRow < nnzPerRow[iR] ? minNNZperRow : nnzPerRow[iR] );
        // Update histogram
        nnzPerRowHist[nnzPerRow[iR]]++;
    }

    // Print the results
    printf( "---------------------------\n" );
    printf( "      NNZ Histogram\n" );
    printf( "---------------------------\n" );
    printf( "      iNNZ       n(NNZ)\n" );
    for( size_t iR = 0, index = 1; iR < nnzPerRowHist.size(); ++iR )
    {
        if( nnzPerRowHist[iR] > 0 ) printf( "%3zu   %3zu     %8d\n", index++, iR, nnzPerRowHist[iR] );
    }
    printf( "---------------------------\n" );
    printf( "NNZ statistics: minima = %d, maxima = %d, average (rounded) = %3.0f\n\n", minNNZperRow, maxNNZperRow,
            static_cast< double >( vecRow.size() ) / nRows );
}
