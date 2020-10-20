#include <functional>
#include <stdexcept>

#include <cufft.h>
#include <cufftXt.h>
#include <cufftdx.hpp>
#include <nvtx3/nvToolsExt.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "block_io.hpp"

#define PRINT 0

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

// *************** FOR NVTX *******************
const uint32_t colors[] { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int      num_colors { sizeof( colors ) / sizeof( uint32_t ) };

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
// *************** FOR NVTX *******************

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

// Complex multiplication
template<typename T>
__device__ T ComplexScale( T const &a, float const &scale ) {
    T c;
    c.x = a.x * scale;
    c.y = a.y * scale;
    return ( c );
}

// Complex multiplication
template<typename T>
__device__ T ComplexMul( T const &a, T const &b ) {
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

// Define variables to point at callbacks
__device__ cufftCallbackLoadC d_loadCallbackPtr   = CB_MulAndScaleInputC;
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_MulAndScaleOutputC;

// Define variables to point at callbacks
__device__ __managed__ cufftCallbackLoadC d_loadManagedCallbackPtr   = CB_MulAndScaleInputC;
__device__ __managed__ cufftCallbackStoreC d_storeManagedCallbackPtr = CB_MulAndScaleOutputC;

// cuFFTDx Forward FFT CUDA kernel
template<class FFT>
__launch_bounds__( FFT::max_threads_per_block ) __global__
    void block_fft_kernel( typename FFT::value_type *inputData, typename FFT::value_type *outputData ) {

    using complex_type = typename FFT::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id { threadIdx.y };

    // Load data from global memory to registers
    example::io<FFT>::load( inputData, thread_data, local_fft_id );

    // Execute FFT
    FFT( ).execute( thread_data, shared_mem );

    // Save results
    example::io<FFT>::store( thread_data, outputData, local_fft_id );
}

// cuFFTDx Inverse FFT CUDA kernel
template<class IFFT, typename T>
__launch_bounds__( IFFT::max_threads_per_block ) __global__
    void block_ifft_kernel( typename IFFT::value_type *inputData,
                            typename IFFT::value_type *outputData,
                            cb_inParams<T> *           inParams,
                            cb_outParams<T> *          outParams ) {

    using complex_type = typename IFFT::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[IFFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id { threadIdx.y };

    // Load data from global memory to registers
    example::io<IFFT>::load( inputData, thread_data, local_fft_id );

    // Execute input callback functionality
    const uint offset = example::io<IFFT>::batch_offset( local_fft_id );
    const uint stride = example::io<IFFT>::stride_size( );
    uint       index  = offset + threadIdx.x;
    for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexScale( ComplexMul( thread_data[i], ( inParams->multiplier )[index] ), inParams->scale );
        index += stride;
    }

    // Execute FFT
    IFFT( ).execute( thread_data, shared_mem );

    // Execute output callback functionality
    index = offset + threadIdx.x;

    for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
        thread_data[i] =
            ComplexScale( ComplexMul( thread_data[i], ( outParams->multiplier )[index] ), outParams->scale );
        index += stride;
    }

    // Save results
    example::io<IFFT>::store( thread_data, outputData, local_fft_id );
}

// Helper function to print
template<typename T>
void printFunction( T *const data ) {
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            printf( "Re = %0.6f; Im = %0.6f\n", data[index( i, kDataSize, j )].x, data[index( i, kDataSize, j )].y );
        }
    }
}

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

// Warm-up function
void warmUpFunction( ) {

    using namespace thrust::placeholders;

    int                        N = 1 << 20;
    thrust::device_vector<int> d_x( N, 2 );  // alloc and copy host to device
    thrust::device_vector<int> d_y( N, 4 );

    // Perform SAXPY on 1M elements
    thrust::transform( d_x.begin( ), d_x.end( ), d_y.begin( ), d_y.begin( ), 2.0f * _1 + _2 );
}

// cuFFT example using explicit memory copies
template<typename T>
void cufftMalloc( T *h_outputData, const int &signalSize, fft_params &fftPlan ) {

    PUSH_RANGE( __FUNCTION__, 1 )

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create host data arrays
    T *h_inputData = new T[signalSize];

    // Create device data arrays
    T *d_inputData;
    T *d_outputData;
    T *d_bufferData;

    PUSH_RANGE( "Prep Input", 2 )
    // Create input data
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_inputData[index( i, kDataSize, j )] = make_cuComplex( ( i + j ), ( i - j ) );
        }
    }

#if PRINT
    printf( "\nPrinting input data\n" );
    printFunction( h_inputData );
#endif

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), signalSize ) );

    // Copy input data to device
    CUDA_RT_CALL( cudaMemcpy( d_inputData, h_inputData, signalSize, cudaMemcpyHostToDevice ) );
    POP_RANGE( )

    PUSH_RANGE( "CB Params", 3 )

    // Create multiplier data
    T *h_multiplier = new T[signalSize];

    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_multiplier[index( i, kDataSize, j )] = make_cuFloatComplex( kMultiplier, kMultiplier );
        }
    }

    T *d_multiplier;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplier ), signalSize ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplier, h_multiplier, signalSize, cudaMemcpyHostToDevice ) );

    // Create callback parameters
    cb_inParams<T> h_inParams;
    h_inParams.scale      = kScale;
    h_inParams.multiplier = d_multiplier;

    // Copy callback parameters to device
    cb_inParams<T> *d_inParams;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<T> ), cudaMemcpyHostToDevice ) );

    cb_outParams<T> h_outParams;
    h_outParams.scale      = kScale;
    h_outParams.multiplier = d_multiplier;

    cb_outParams<T> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<T> ), cudaMemcpyHostToDevice ) );

    POP_RANGE( )

    PUSH_RANGE( "cufftPlanMany", 4 )
    CUDA_RT_CALL( cufftPlanMany( &fft_forward,
                                 fftPlan.rank,
                                 fftPlan.n,
                                 fftPlan.inembed,
                                 fftPlan.istride,
                                 fftPlan.idist,
                                 fftPlan.onembed,
                                 fftPlan.ostride,
                                 fftPlan.odist,
                                 CUFFT_C2C,
                                 fftPlan.batch ) );
    CUDA_RT_CALL( cufftPlanMany( &fft_inverse,
                                 fftPlan.rank,
                                 fftPlan.n,
                                 fftPlan.inembed,
                                 fftPlan.istride,
                                 fftPlan.idist,
                                 fftPlan.onembed,
                                 fftPlan.ostride,
                                 fftPlan.odist,
                                 CUFFT_C2C,
                                 fftPlan.batch ) );
    POP_RANGE( )

    PUSH_RANGE( "CB Pointers", 5 )
    // Create host callback pointers
    cufftCallbackLoadC  h_loadCallbackPtr;
    cufftCallbackStoreC h_storeCallbackPtr;

    // Copy device pointers to host
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );
    POP_RANGE( )

    PUSH_RANGE( "cufftXtSetCallback", 6 )
    // Set input callback
    CUDA_RT_CALL(
        cufftXtSetCallback( fft_inverse, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&d_outParams ) );
    POP_RANGE( )

    PUSH_RANGE( "cufftExecC2C", 7 )
    // Execute FFT plan
    CUDA_RT_CALL( cufftExecC2C( fft_forward, d_inputData, d_bufferData, CUFFT_FORWARD ) );

#if PRINT
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_bufferData, signalSize, cudaMemcpyDeviceToHost ) );
    printf( "\nPrinting buffer data\n" );
    printFunction( h_outputData );
#endif

    CUDA_RT_CALL( cufftExecC2C( fft_inverse, d_bufferData, d_outputData, CUFFT_INVERSE ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    POP_RANGE( )

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );

#if PRINT
    printf( "\nPrinting output data\n" );
    printFunction( h_outputData );
#endif

    // Cleanup Memory
    delete[]( h_inputData );
    delete[]( h_multiplier );
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );
    CUDA_RT_CALL( cudaFree( d_multiplier ) );

    POP_RANGE( )
}

// cuFFT example using managed memory copies
template<typename T>
void cufftManaged( T *h_outputData, const int &signalSize, fft_params &fftPlan ) {

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    PUSH_RANGE( __FUNCTION__, 1 )

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create data arrays
    T *inputData;
    T *outputData;
    T *bufferData;

    PUSH_RANGE( "Prep Input", 2 )
    CUDA_RT_CALL( cudaMallocManaged( &inputData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &outputData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &bufferData, signalSize ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Create input data
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            inputData[i * kDataSize + j] = make_cuComplex( ( i + j ), ( i - j ) );
        }
    }

    POP_RANGE( )

#if PRINT
    printf( "\nPrinting input data\n" );
    printFunction<T>( inputData );
#endif

    PUSH_RANGE( "CB Params", 3 )

    // Create callback parameters
    cb_inParams<T> *h_inParams;

    CUDA_RT_CALL( cudaMallocManaged( &h_inParams, sizeof( cb_inParams<T> ) ) );
    h_inParams->scale = kScale;
    CUDA_RT_CALL( cudaMallocManaged( &h_inParams->multiplier, signalSize ) );

    // Create multiplier data
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_inParams->multiplier[index( i, kDataSize, j )] = make_cuComplex( kMultiplier, kMultiplier );
        }
    }

    cb_outParams<T> *h_outParams;

    CUDA_RT_CALL( cudaMallocManaged( &h_outParams, sizeof( cb_outParams<T> ) ) );
    h_outParams->scale = kScale;
    CUDA_RT_CALL( cudaMallocManaged( &h_outParams->multiplier, signalSize ) );
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_outParams->multiplier[index( i, kDataSize, j )] = h_inParams->multiplier[index( i, kDataSize, j )];
        }
    }

    POP_RANGE( )

    PUSH_RANGE( "cufftPlanMany", 4 )
    CUDA_RT_CALL( cufftPlanMany( &fft_forward,
                                 fftPlan.rank,
                                 fftPlan.n,
                                 fftPlan.inembed,
                                 fftPlan.istride,
                                 fftPlan.idist,
                                 fftPlan.onembed,
                                 fftPlan.ostride,
                                 fftPlan.odist,
                                 CUFFT_C2C,
                                 fftPlan.batch ) );
    CUDA_RT_CALL( cufftPlanMany( &fft_inverse,
                                 fftPlan.rank,
                                 fftPlan.n,
                                 fftPlan.inembed,
                                 fftPlan.istride,
                                 fftPlan.idist,
                                 fftPlan.onembed,
                                 fftPlan.ostride,
                                 fftPlan.odist,
                                 CUFFT_C2C,
                                 fftPlan.batch ) );
    POP_RANGE( )

    PUSH_RANGE( "cufftXtSetCallback", 6 )
    // Set input callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&d_loadManagedCallbackPtr, CUFFT_CB_LD_COMPLEX, ( void ** )&h_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&d_storeManagedCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&h_outParams ) );
    POP_RANGE( )

    PUSH_RANGE( "cufftExecC2C", 7 )
    // Execute FFT plan
    CUDA_RT_CALL( cufftExecC2C( fft_forward, inputData, bufferData, CUFFT_FORWARD ) );

#if PRINT
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, bufferData, signalSize, cudaMemcpyDeviceToHost ) );
    printf( "\nPrinting buffer data\n" );
    printFunction( h_outputData );
#endif

    CUDA_RT_CALL( cufftExecC2C( fft_inverse, bufferData, outputData, CUFFT_INVERSE ) );

    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    POP_RANGE( )

#if PRINT
    printf( "\nPrinting output data\n" );
    printFunction<cufftComplex>( outputData );
#endif

    CUDA_RT_CALL( cudaMemcpy( h_outputData, outputData, signalSize, cudaMemcpyDeviceToHost ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( inputData ) );
    CUDA_RT_CALL( cudaFree( outputData ) );
    CUDA_RT_CALL( cudaFree( bufferData ) );
    CUDA_RT_CALL( cudaFree( h_inParams->multiplier ) );
    CUDA_RT_CALL( cudaFree( h_inParams ) );
    CUDA_RT_CALL( cudaFree( h_outParams->multiplier ) );
    CUDA_RT_CALL( cudaFree( h_outParams ) );

    POP_RANGE( )
}

template<uint A, typename T>
void cuFFTDxMalloc( T *h_outputData ) {

    PUSH_RANGE( __FUNCTION__, 1 )

    PUSH_RANGE( "Prep Input", 2 )

    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    using FFT = decltype( cufftdx::Block( ) + cufftdx::Size<kDataSize>( ) + cufftdx::Type<cufftdx::fft_type::c2c>( ) +
                          cufftdx::Direction<cufftdx::fft_direction::forward>( ) + cufftdx::Precision<float>( ) +
                          cufftdx::ElementsPerThread<kElementsPerThread>( ) + cufftdx::FFTsPerBlock<kBatch>( ) +
                          cufftdx::SM<A>( ) );

    using IFFT = decltype( cufftdx::Block( ) + cufftdx::Size<kDataSize>( ) + cufftdx::Type<cufftdx::fft_type::c2c>( ) +
                           cufftdx::Direction<cufftdx::fft_direction::inverse>( ) + cufftdx::Precision<float>( ) +
                           cufftdx::ElementsPerThread<kElementsPerThread>( ) + cufftdx::FFTsPerBlock<kBatch>( ) +
                           cufftdx::SM<A>( ) );

    using complex_type = typename FFT::value_type;

    // Allocate managed memory for input/output
    auto size { FFT::ffts_per_block * cufftdx::size_of<FFT>::value };  // cufftdx::Size<5>() *
                                                                       // cufftdx::FFTsPerBlock<1>()
    auto sizeBytes { size * sizeof( complex_type ) };                  // Should be same as signalSize

    complex_type *h_inputData = new complex_type[sizeBytes];

    // Create data
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_inputData[index( i, kDataSize, j )] = complex_type { float( i + j ), float( i - j ) };
        }
    }

#if PRINT
    printf( "\nPrinting input data\n" );
    printFunction( h_inputData );
#endif

    // Create data arrays and allocate
    complex_type *d_inputData;
    complex_type *d_outputData;
    complex_type *d_bufferData;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), sizeBytes ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), sizeBytes ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), sizeBytes ) );

    // Copy input data to device
    CUDA_RT_CALL( cudaMemcpy( d_inputData, h_inputData, sizeBytes, cudaMemcpyHostToDevice ) );

    POP_RANGE( )

    PUSH_RANGE( "CB Params", 3 )

    // Create multiplier data
    complex_type *h_multiplier = new complex_type[sizeBytes];

    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_multiplier[index( i, kDataSize, j )] = complex_type { kMultiplier, kMultiplier };
        }
    }

    complex_type *d_multiplier;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplier ), sizeBytes ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplier, h_multiplier, sizeBytes, cudaMemcpyHostToDevice ) );

    // Create callback parameters
    cb_inParams<complex_type> h_inParams;
    h_inParams.scale      = kScale;
    h_inParams.multiplier = d_multiplier;

    // Copy callback parameters to device
    cb_inParams<complex_type> *d_inParams;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<complex_type> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<complex_type> ), cudaMemcpyHostToDevice ) );

    cb_outParams<complex_type> h_outParams;
    h_outParams.scale      = kScale;
    h_outParams.multiplier = d_multiplier;

    cb_outParams<complex_type> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<complex_type> ) ) );
    CUDA_RT_CALL(
        cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<complex_type> ), cudaMemcpyHostToDevice ) );

    POP_RANGE( )

    PUSH_RANGE( "cufftExecC2C", 7 )

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>( d_inputData, d_bufferData );
    CUDA_RT_CALL( cudaPeekAtLastError( ) );

#if PRINT
    // Copy data from device to host
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_bufferData, sizeBytes, cudaMemcpyDeviceToHost ) );
    printf( "\nPrinting buffer data\n" );
    printFunction( h_outputData );
#endif

    block_ifft_kernel<IFFT, complex_type>
        <<<1, FFT::block_dim, FFT::shared_memory_size>>>( d_bufferData, d_outputData, d_inParams, d_outParams );
    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    POP_RANGE( )

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, sizeBytes, cudaMemcpyDeviceToHost ) );

#if PRINT
    printf( "\nPrinting output data\n" );
    printFunction( h_outputData );
#endif

    // Cleanup Memory
    delete[]( h_inputData );
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );

    POP_RANGE( )
}

// Returns CUDA device compute capability
uint get_cuda_device_arch( ) {
    int device;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    cudaDeviceProp props;
    CUDA_RT_CALL( cudaGetDeviceProperties( &props, device ) );

    return ( static_cast<uint>( props.major ) * 100 + static_cast<uint>( props.minor ) * 10 );
}

int main( int argc, char **argv ) {

    // Calculate size of signal array to process
    const size_t signalSize { sizeof( cufftComplex ) * kDataSize * kBatch };

    // Set fft plan parameters
    fft_params fftPlan { kRank, { kDataSize }, 1, 1, kDataSize, kDataSize, { 0 }, { 0 }, kBatch };

    cufftComplex *cufftHostData        = new cufftComplex[signalSize];
    cufftComplex *cufftManagedHostData = new cufftComplex[signalSize];
    cufftComplex *cufftDxHostData      = new cufftComplex[signalSize];

    // Warm-up GPU
    warmUpFunction( );

    // Run basic cuFFT example with callbacks
    cufftMalloc<cufftComplex>( cufftHostData, signalSize, fftPlan );

#if PRINT
    printf( "\nPrinting cufftHostData data\n" );
    printFunction( cufftHostData );
#endif

    cufftManaged<cufftComplex>( cufftManagedHostData, signalSize, fftPlan );

#if PRINT
    printf( "\nPrinting cufftManagedHostData data\n" );
    printFunction( cufftManagedHostData );
#endif

    // Verify cuFFT (cudaMalloc vs cudaManagedMalloc) have the same results
    printf( "Compare cuFFT (cudaMalloc vs cudaMallocManaged)\n" );
    verifyResults( cufftHostData, cufftManagedHostData, signalSize );

    // Retrieve GPU architecture
    const uint arch { get_cuda_device_arch( ) };

    // Run cuFFTDx example to replicate cuFFT functionality
    switch ( arch ) {
    case 750:
        cuFFTDxMalloc<750, cufftComplex>( cufftDxHostData );
#if PRINT
        printf( "\nPrinting cufftDxHostData data\n" );
        printFunction( cufftDxHostData );
#endif
        break;
    default:
        printf( "GPU architecture must be 7.0 or greater to use cuFFTDx\n "
                "Skipping Test!\n" );
        break;
    }

    // Verify cuFFT and cuFFTDx have the same results
    printf( "Compare cuFFT and cuFFTDx (cudaMalloc)\n" );
    verifyResults( cufftHostData, cufftDxHostData, signalSize );

    delete[]( cufftHostData );
    delete[]( cufftManagedHostData );
    delete[]( cufftDxHostData );
    CUDA_RT_CALL( cudaDeviceReset( ) );
}
