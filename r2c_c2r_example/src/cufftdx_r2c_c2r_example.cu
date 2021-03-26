#include <functional>
#include <random>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cufftMalloc_r2r.h"
#include "cufftManaged_r2r.h"
#include "cufftdxMalloc_r2r.h"

#include "../../common/cuda_helper.h"

// Warm-up function
void warmUpFunction( ) {

    using namespace thrust::placeholders;

    int N = 1 << 20;

    thrust::device_vector<int> d_x( N, 2 );  // alloc and copy host to device
    thrust::device_vector<int> d_y( N, 4 );

    // Perform SAXPY on 1M elements
    for ( int i = 0; i < 1024; i++ )
        thrust::transform( d_x.begin( ), d_x.end( ), d_y.begin( ), d_y.begin( ), 2 * _1 + _2 );
}

// Returns CUDA device compute capability
uint get_cuda_device_arch( ) {
    int device;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    cudaDeviceProp props;
    CUDA_RT_CALL( cudaGetDeviceProperties( &props, device ) );

    return ( static_cast<uint>( props.major ) * 100 + static_cast<uint>( props.minor ) * 10 );
}

template<uint ARCH, uint SIZE, uint BATCH, uint FPB, uint EPT>
void benchmark_r2r( ) {

#ifdef USE_DOUBLE
    using run_type   = double;
    using cufft_type = cufftDoubleReal;
    using buf_type   = cufftDoubleComplex;
#else
    using run_type   = float;
    using cufft_type = cufftReal;
    using buf_type   = cufftComplex;
#endif

    // Calculate size of signal array to process
    const size_t signalSize { sizeof( cufft_type ) * SIZE * BATCH };

    // Set fft plan parameters
    fft_params fftPlan { kRank, { SIZE }, 1, 1, SIZE, SIZE, { 0 }, { 0 }, BATCH };

    cufft_type *cufftHostData        = new cufft_type[signalSize];
    cufft_type *cufftManagedHostData = new cufft_type[signalSize];
    cufft_type *cufftDxHostData      = new cufft_type[signalSize];

    // Create input signal
    cufft_type *inputData = new cufft_type[signalSize];

    std::mt19937                             eng;
    std::uniform_real_distribution<run_type> dist( kLower, kUpper );
    // printf("\n[");
    for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        inputData[i] = temp;
        // printf("%f, ", temp);
    }
    // printf("]\n");

    // Create multipler signal
    buf_type *  multDataIn  = new buf_type[signalSize];
    cufft_type *multDataOut = new cufft_type[signalSize * 2];
    for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        multDataIn[i].x = temp;
        multDataIn[i].y = temp + 1;
        multDataOut[i]  = temp;
        // printf("%d: %f\n", i, temp);
    }

    run_type scalar { 1.7 };

    std::printf( "FFT Size: %d -- Batch: %d -- FFT Per Block: %d -- EPT: %d\n", SIZE, BATCH, FPB, EPT );
    cufftMalloc_r2r<cufft_type, buf_type, run_type, SIZE, BATCH>(
        inputData, multDataIn, multDataOut, scalar, signalSize, fftPlan, cufftHostData );

    cufftManaged_r2r<cufft_type, buf_type, run_type, SIZE, BATCH>(
        inputData, multDataIn, multDataOut, scalar, signalSize, fftPlan, cufftManagedHostData );
    verifyResults_r2r<cufft_type, SIZE, BATCH>( cufftHostData, cufftManagedHostData, signalSize );

    cufftdxMalloc_r2r<cufft_type, buf_type, run_type, ARCH, SIZE, BATCH, FPB, EPT>(
        inputData, multDataIn, multDataOut, scalar, signalSize, cufftDxHostData );

    // // Verify cuFFT and cuFFTDx have the same results
    verifyResults_r2r<cufft_type, SIZE, BATCH>( cufftHostData, cufftDxHostData, signalSize );

    delete[]( inputData );
    delete[]( multDataIn );
    delete[]( multDataOut );
    delete[]( cufftHostData );
    delete[]( cufftManagedHostData );
    delete[]( cufftDxHostData );
}

int main( int argc, char **argv ) {

    // Retrieve GPU architecture
    const uint arch { get_cuda_device_arch( ) };

    // Warm-up GPU
    warmUpFunction( );

    switch ( arch ) {
        // template<uint ARCH, uint SIZE, uint BATCH, uint FPB, uint EPT>
#ifdef USE_DOUBLE
    case 700:
        benchmark_r2r<700, 8192, 16384, 1, 16>( );
        break;
    case 750:
        benchmark_r2r<750, 2048, 16384, 1, 16>( );
        break;
    case 800:
        benchmark_r2r<800, 16384, 16384, 1, 16>( );
        break;
    default:
        printf( "GPU architecture must be 7.0 or greater to use cuFFTDx\n "
                "Skipping Test!\n" );
        break;
    }
#else
    case 700:
        benchmark_r2r<700, 16384, 16384, 1, 32>( );
        break;
    case 750:
        benchmark_r2r<750, 4096, 16384, 1, 16>( );
        break;
    case 800:
        benchmark_r2r<800, 32768, 16384, 1, 32>( );
        break;
    default:
        printf( "GPU architecture must be 7.0 or greater to use cuFFTDx\n "
                "Skipping Test!\n" );
        break;
    }
#endif

    CUDA_RT_CALL( cudaDeviceReset( ) );
}
