#pragma once



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

// ***************** FOR NVTX MARKERS *******************
#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

const uint32_t colors[]   = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int      num_colors = sizeof( colors ) / sizeof( uint32_t );

#define PUSH_RANGE( name, cid )                                                                                        \
    {                                                                                                                  \
        int color_id                      = cid;                                                                       \
        color_id                          = color_id % num_colors;                                                     \
        nvtxEventAttributes_t eventAttrib = { 0 };                                                                     \
        eventAttrib.version               = NVTX_VERSION;                                                              \
        eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                             \
        eventAttrib.colorType             = NVTX_COLOR_ARGB;                                                           \
        eventAttrib.color                 = colors[color_id];                                                          \
        eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;                                                   \
        eventAttrib.message.ascii         = name;                                                                      \
        nvtxRangePushEx( &eventAttrib );                                                                               \
    }
#define POP_RANGE( ) nvtxRangePop( );
#else
#define PUSH_RANGE( name, cid )
#define POP_RANGE( )
#endif
// ***************** FOR NVTX MARKERS *******************

constexpr int   kDataSize { 4096 };
constexpr int   kBatch { 1 };
constexpr int   kRank { 1 };
constexpr int   kElementsPerThread { 8 };
constexpr float kScale { 1.0f };
constexpr float kMultiplier { 1.0f };
constexpr float kTolerance { 1e-3f };  // Compare cuFFT / cuFFTDx results

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
__device__ T CB_MulAndScaleInputC( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr ) {
    cb_inParams<T> *params = static_cast<cb_inParams<T> *>( callerInfo );
    return ( ComplexScale( ComplexMul( static_cast<T *>( dataIn )[offset], ( params->multiplier )[offset] ),
                           params->scale ) );
}

// Output Callback
template<typename T>
__device__ void CB_MulAndScaleOutputC( void *dataOut, size_t offset, T element, void *callerInfo, void *sharedPtr ) {
    cb_outParams<T> *params { static_cast<cb_outParams<T> *>( callerInfo ) };

    static_cast<T *>( dataOut )[offset] =
        ComplexScale( ComplexMul( element, ( params->multiplier )[offset] ), params->scale );
}

// ******************* FOR PRINTING *********************
#ifdef PRINT
#include <cstring>

template<typename T>
void printFunction( std::string const str, T *const data ) {
    printf( "\n%s\n", str.c_str() );
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            printf( "Re = %0.6f; Im = %0.6f\n", data[index( i, kDataSize, j )].x, data[index( i, kDataSize, j )].y );
        }
    }
}
#else
template<typename T>
void printFunction( std::string const str, T *const data ) {}
#endif
// ******************* FOR PRINTING *********************

template<typename T>
void verifyResults( T const *ref, T const *alt, const int &signalSize ) {

    float2 *relError = new float2[signalSize];
    int     counter {};

    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            int idx         = index( i, kDataSize, j );
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