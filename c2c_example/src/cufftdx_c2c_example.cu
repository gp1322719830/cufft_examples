#include <functional>
#include <random>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cufftMalloc_c2c.h"
#include "cufftManaged_c2c.h"
#include "cufftdxMalloc_c2c.h"

#include "../../common/cuda_helper.h"

template<uint ARCH, uint SIZE, uint BATCH, uint FPB, uint EPT>
void benchmark( ) {

#ifdef USE_DOUBLE
    using run_type   = double;
    using cufft_type = cufftDoubleComplex;
#else
    using run_type   = float;
    using cufft_type = cufftComplex;
#endif

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    // Calculate size of signal array to process
    const size_t signalSize { sizeof( cufft_type ) * SIZE * BATCH };

    // Set fft plan parameters
    fft_params fftPlan { kRank, { SIZE }, 1, 1, SIZE, SIZE, { 0 }, { 0 }, BATCH };

    cufft_type *cufftHostData;
    cufft_type *cufftManagedHostData;
    cufft_type *cufftDxHostData;

    CUDA_RT_CALL( cudaMallocManaged( &cufftHostData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &cufftManagedHostData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &cufftDxHostData, signalSize ) );

    // Create input signal
    cufft_type *inputData;
    CUDA_RT_CALL( cudaMallocManaged( &inputData, signalSize ) );

    std::mt19937                             eng;
    std::uniform_real_distribution<run_type> dist( kLower, kUpper );
    for ( int i = 0; i < ( 2 * SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        inputData[i].x = temp;
        inputData[i].y = temp;
    }

    // Create multipler signal
    // cufft_type *multData = new cufft_type[SIZE * BATCH * 2];
    cufft_type *multData;
    CUDA_RT_CALL( cudaMallocManaged( &multData, signalSize ) );
    for ( int i = 0; i < ( 2 * SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        multData[i].x = temp;
        multData[i].y = temp;
    }

    run_type scalar { 1.7 };

    std::printf( "FFT Size: %d -- Batch: %d -- FFT Per Block: %d -- EPT: %d\n", SIZE, BATCH, FPB, EPT );
    cufftMalloc<cufft_type, run_type, SIZE, BATCH>(
        device, inputData, multData, scalar, signalSize, fftPlan, cufftHostData );

    cufftManaged<cufft_type, run_type, SIZE, BATCH>(
        device, inputData, multData, scalar, signalSize, fftPlan, cufftManagedHostData );
    verifyResults_c2c<cufft_type, SIZE, BATCH>( cufftHostData, cufftManagedHostData, SIZE );

    cufftdxMalloc<cufft_type, run_type, ARCH, SIZE, BATCH, FPB, EPT>(
        device, inputData, multData, scalar, signalSize, cufftDxHostData );

    // Verify cuFFT and cuFFTDx have the same results
    verifyResults_c2c<cufft_type, SIZE, BATCH>( cufftHostData, cufftDxHostData, SIZE );
}

int main( int argc, char **argv ) {

    // Retrieve GPU architecture
    const uint arch { get_cuda_device_arch( ) };

    switch ( arch ) {
        // template<uint ARCH, uint SIZE, uint BATCH, uint FPB, uint EPT>
#ifdef USE_DOUBLE
    case 700:
        benchmark<700, 8192, 16384, 1, 16>( );
        break;
    case 750:
        benchmark<750, 2048, 16384, 1, 16>( );
        break;
    case 800:
        benchmark<800, 16384, 16384, 1, 16>( );
        break;
    default:
        std::printf( "GPU architecture not found see cuFFTDx docs\n "
                     "Skipping Test!\n" );
        break;
    }
#else
    case 700:
        benchmark<700, 16384, 16384, 1, 32>( );
        break;
    case 750:
        benchmark<750, 4096, 16384, 1, 16>( );
        break;
    case 800:
        benchmark<800, 32768, 16384, 1, 32>( );
        break;
    default:
        std::printf( "GPU architecture not found see cuFFTDx docs\n "
                     "Skipping Test!\n" );
        break;
    }
#endif

    CUDA_RT_CALL( cudaDeviceReset( ) );
}
