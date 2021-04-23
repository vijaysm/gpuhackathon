#ifndef __SPMVAPP_UTILS_H__
#define __SPMVAPP_UTILS_H__

// C++ includes
#include <iostream>
#include <chrono>

// Some global typedefs
typedef int MOABSInt;
typedef size_t MOABUInt;
typedef double MOABReal;

class SpMVOperator
{
  public:
    virtual ~SpMVOperator(){};
    virtual void CreateOperator( const std::vector< MOABSInt >& vecRow, const std::vector< MOABSInt >& vecCol,
                                 const std::vector< MOABReal >& vecS )           = 0;
    virtual bool PerformVerification( const std::vector< MOABReal >& vecAreasA,
                                      const std::vector< MOABReal >& vecAreasB ) = 0;
    virtual void PerformSpMV( int n_remap_iterations = 1 )                       = 0;
    virtual void PerformSpMVTranspose( int n_remap_iterations = 1 )              = 0;

    bool is_initialized()
    {
        return initialized;
    }

  protected:
    SpMVOperator( MOABSInt nRows, MOABSInt nCols, MOABSInt nNNZs, MOABSInt nVecs, bool requireTranspose = false )
        : nOpRows( nRows ), nOpCols( nCols ), nOpNNZs( nNNZs ), nRHSV( nVecs ), initialized( false ),
          enableTransposeOp( requireTranspose ){};

    const MOABSInt nOpRows, nOpCols, nOpNNZs, nRHSV;
    bool initialized, enableTransposeOp;
};

// Utility timer class
struct HighresTimer
{
  public:
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::high_resolution_clock::time_point Timer;

    HighresTimer() {}

    void push()
    {
        start = Clock::now();
    }

    void pop( const std::string& eventName, bool printLog = false )
    {
        std::chrono::nanoseconds elapsed =
            std::chrono::duration_cast< std::chrono::nanoseconds >( Clock::now() - start );
        timeLog[eventName] = elapsed;
        if( printLog ) this->print( eventName );
    }

    void print( const std::string& eventName )
    {
        std::cout << "[ " << eventName << " ]: elapsed = " << static_cast< double >( timeLog[eventName].count() / 1e6 )
                  << " milli-seconds" << std::endl;
    }

    double elapsed( const std::string& eventName )
    {
        return timeLog[eventName].count();
    }

    Timer start;
    std::map< std::string, std::chrono::nanoseconds > timeLog;
};

#endif  // __SPMVAPP_UTILS_H__