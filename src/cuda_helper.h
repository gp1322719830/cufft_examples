#pragma once

#include <cuda.h>

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************


// ***************** TIMER *******************
class Timer {

  public:
    // GPU Timer
    void startGPUTimer( ) {
        cudaEventCreate( &startEvent, cudaEventBlockingSync );
        cudaEventRecord( startEvent );
    }  // startGPUTimer

    void stopGPUTimer( ) {
        cudaEventCreate( &stopEvent, cudaEventBlockingSync );
        cudaEventRecord( stopEvent );
        cudaEventSynchronize( stopEvent );
    }  // stopGPUTimer

    void printGPUTime( ) {
        cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
        std::printf( "%0.2f ms\n", elapsed_gpu_ms );
    }  // printGPUTime

    void printGPUTime( int const &loops ) {
        cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
        std::printf( "%0.2f ms\n", elapsed_gpu_ms / loops );
    }  // printGPUTime

    void stopAndPrintGPU( ) {
        stopGPUTimer( );
        printGPUTime( );
    }

    void stopAndPrintGPU( int const &loops ) {
        stopGPUTimer( );
        printGPUTime( loops );
    }

    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};
};
// ***************** TIMER *******************

constexpr int   kLoops { 1024 };
constexpr int   kRank { 1 };
constexpr float kScale { 1.7f };
// constexpr float kMultiplier { 1.3f };
constexpr float kTolerance { 1e-2f };  // Compare cuFFT / cuFFTDx results

constexpr int index( int i, int j, int k ) {
    return ( i * j + k );
}

typedef struct _fft_params {
    int rank;            // 1D FFTs
    int n[kRank];        // Size of the Fourier transform
    int istride;         // Distance between two successive input elements
    int ostride;         // Distance between two successive output elements
    int idist;           // Distance between input batches
    int odist;           // Distance between output batches
    int inembed[kRank];  // Input size with pitch (ignored for 1D transforms)
    int onembed[kRank];  // Output size with pitch (ignored for 1D transforms)
    int batch;           // Number of batched executions
} fft_params;

template<typename T>
struct cb_inParams {
    T *   multiplier;
    float scale;
};

template<typename T>
struct cb_outParams {
    T *   multiplier;
    float scale;
};

// Complex multiplication
template<typename T>
__device__ T ComplexScale( T const &a, float const &scale ) {
    T c;
    c.x = a.x * scale;
    c.y = a.y * scale;
    return ( c );
}

// Complex multiplication
template<typename T, typename U>
__device__ T ComplexMul( T const &a, U const &b ) {
    T c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return ( c );
}

// Input Callback
template<typename T>
__device__ T CB_MulAndScaleInput( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr ) {
    cb_inParams<T> *params = static_cast<cb_inParams<T> *>( callerInfo );
    return ( ComplexScale( ComplexMul( static_cast<T *>( dataIn )[offset], ( params->multiplier )[offset] ),
                           params->scale ) );
}

// Output Callback
template<typename T>
__device__ void CB_MulAndScaleOutput( void *dataOut, size_t offset, T element, void *callerInfo, void *sharedPtr ) {
    cb_outParams<T> *params { static_cast<cb_outParams<T> *>( callerInfo ) };

    static_cast<T *>( dataOut )[offset] =
        ComplexScale( ComplexMul( element, ( params->multiplier )[offset] ), params->scale );
}

#ifdef PRINT
template<typename T, uint SIZE, uint BATCH>
void verifyResults( T const *ref, T const *alt, const size_t &signalSize ) {

    printf( "\nCompare results\n" );

    float2 *relError = new float2[signalSize];
    int     counter {};

    for ( int i = 0; i < BATCH; i++ ) {
        for ( int j = 0; j < SIZE; j++ ) {
            size_t idx      = index( i, SIZE, j );
            relError[idx].x = ( ref[idx].x - alt[idx].x ) / ref[idx].x;
            relError[idx].y = ( ref[idx].y - alt[idx].y ) / ref[idx].y;

            if ( relError[idx].x > kTolerance ) {
                printf( "R - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                        i,
                        j,
                        ref[idx].x,
                        alt[idx].x,
                        relError[idx].x,
                        kTolerance );
                counter++;
            }

            if ( relError[idx].y > kTolerance ) {
                printf( "I - Batch %d: Element %d: %f - %f (%0.7f) > %f\n",
                        i,
                        j,
                        ref[idx].y,
                        alt[idx].y,
                        relError[idx].y,
                        kTolerance );
                counter++;
            }
        }
    }

    if ( !counter ) {
        printf( "All values match!\n\n" );
    }
}
#else
template<typename T, uint SIZE, uint BATCH>
void verifyResults( T const *ref, T const *alt, const size_t &signalSize ) {}
#endif
