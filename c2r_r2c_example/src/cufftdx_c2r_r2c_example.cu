#include <functional>
#include <random>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cufftMalloc_c2r_r2c.h"
#include "cufftManaged_c2r_r2c.h"
#include "cufftdxMalloc_c2r_r2c.h"

#include "../../common/cuda_helper.h"

template<uint ARCH, uint SIZE, uint BATCH, uint FPB, uint EPT>
void benchmark( ) {

#ifdef USE_DOUBLE
    using run_type   = double;
    using cufft_type = cufftDoubleComplex;
    using buf_type   = cufftDoubleReal;
#else
    using run_type   = float;
    using cufft_type = cufftComplex;
    using buf_type   = cufftReal;
#endif

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    // Calculate size of signal array to process
    const size_t signalSize { sizeof( cufft_type ) * SIZE * BATCH };
    const size_t bufferSize { sizeof( buf_type ) * SIZE * BATCH };

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
    for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        inputData[i].x = temp;
        inputData[i].y = temp;
    }

    // Create multipler signal
    buf_type *  multDataIn;
    cufft_type *multDataOut;

    CUDA_RT_CALL( cudaMallocManaged( &multDataIn, bufferSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &multDataOut, signalSize ) );
    
    for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        multDataIn[i]    = temp;
        multDataOut[i].x = temp;
        multDataOut[i].y = temp + 1;
    }

    run_type scalar { 1.7 };

    std::printf( "FFT Size: %d -- Batch: %d -- FFT Per Block: %d -- EPT: %d\n", SIZE, BATCH, FPB, EPT );
    cufftMalloc<cufft_type, buf_type, run_type, SIZE, BATCH>(
        device, inputData, multDataIn, multDataOut, scalar, signalSize, bufferSize, fftPlan, cufftHostData );

    cufftManaged<cufft_type, buf_type, run_type, SIZE, BATCH>(
        device, inputData, multDataIn, multDataOut, scalar, signalSize, bufferSize, fftPlan, cufftManagedHostData );
    verifyResults_c2c<cufft_type, SIZE, BATCH>( cufftHostData, cufftManagedHostData, ( SIZE / 2 + 1 ) );

    cufftdxMalloc<cufft_type, buf_type, run_type, ARCH, SIZE, BATCH, FPB, EPT>(
        device, inputData, multDataIn, multDataOut, scalar, signalSize, bufferSize, cufftDxHostData );

    // Verify cuFFT and cuFFTDx have the same results
    verifyResults_c2c<cufft_type, SIZE, BATCH>( cufftHostData, cufftDxHostData, ( SIZE / 2 + 1 ) );

    CUDA_RT_CALL( cudaFree( inputData ) );
    CUDA_RT_CALL( cudaFree( multDataIn ) );
    CUDA_RT_CALL( cudaFree( multDataOut ) );
    CUDA_RT_CALL( cudaFree( cufftHostData ) );
    CUDA_RT_CALL( cudaFree( cufftManagedHostData ) );
    CUDA_RT_CALL( cudaFree( cufftDxHostData ) );
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
