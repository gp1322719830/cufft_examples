#include <functional>
#include <random>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cufftMalloc_r2c_c2r.h"
#include "cufftManaged_r2c_c2r.h"
#include "cufftdxMalloc_r2c_c2r.h"

#include "../../common/cuda_helper.h"

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
    for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        inputData[i] = temp;
    }

    // Create multipler signal
    buf_type *  multDataIn  = new buf_type[signalSize];
    cufft_type *multDataOut = new cufft_type[signalSize * 2];
    for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
        run_type temp { dist( eng ) };
        multDataIn[i].x = temp;
        multDataIn[i].y = temp + 1;
        multDataOut[i]  = temp;
    }

    run_type scalar { 1.7 };

    std::printf( "FFT Size: %d -- Batch: %d -- FFT Per Block: %d -- EPT: %d\n", SIZE, BATCH, FPB, EPT );
    cufftMalloc_r2r<cufft_type, buf_type, run_type, SIZE, BATCH>(
        inputData, multDataIn, multDataOut, scalar, signalSize, fftPlan, cufftHostData );

    cufftManaged_r2r<cufft_type, buf_type, run_type, SIZE, BATCH>(
        inputData, multDataIn, multDataOut, scalar, signalSize, fftPlan, cufftManagedHostData );
    verifyResults_r2r<cufft_type, SIZE, BATCH>( cufftHostData, cufftManagedHostData, SIZE );

    cufftdxMalloc_r2r<cufft_type, buf_type, run_type, ARCH, SIZE, BATCH, FPB, EPT>(
        inputData, multDataIn, multDataOut, scalar, signalSize, cufftDxHostData );

    // // Verify cuFFT and cuFFTDx have the same results
    verifyResults_r2r<cufft_type, SIZE, BATCH>( cufftHostData, cufftDxHostData, SIZE );

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
        std::printf( "GPU architecture not found see cuFFTDx docs\n "
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
        std::printf( "GPU architecture not found see cuFFTDx docs\n "
                     "Skipping Test!\n" );
        break;
    }
#endif

    CUDA_RT_CALL( cudaDeviceReset( ) );
}
