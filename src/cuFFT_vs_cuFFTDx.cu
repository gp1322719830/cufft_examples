#include <functional>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "cufftMalloc.h"
#include "cufftManaged.h"
#include "cufftdxMalloc.h"

// Warm-up function
void warmUpFunction( ) {

    using namespace thrust::placeholders;

    int N = 1 << 20;

    thrust::device_vector<int> d_x( N, 2 );  // alloc and copy host to device
    thrust::device_vector<int> d_y( N, 4 );

    // Perform SAXPY on 1M elements
    for (int i = 0; i < 1024; i++)
        thrust::transform( d_x.begin( ), d_x.end( ), d_y.begin( ), d_y.begin( ), 2.0f * _1 + _2 );
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
void benchmark() {
    // Calculate size of signal array to process
    const size_t signalSize { sizeof( cufftComplex ) * SIZE * BATCH };

    // Set fft plan parameters
    fft_params fftPlan { kRank, { SIZE }, 1, 1, SIZE, SIZE, { 0 }, { 0 }, BATCH };

    cufftComplex *cufftHostData        = new cufftComplex[signalSize];
    cufftComplex *cufftManagedHostData = new cufftComplex[signalSize];
    cufftComplex *cufftDxHostData      = new cufftComplex[signalSize];

    // float *h_inputData = new complex_type[sizeBytes * 2];
    // std::mt19937 eng;
    // std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    // for ( int i = 0; i < BATCH; i++ ) {
    //     for ( int j = 0; j < SIZE; j++ ) {
    //         float temp { dist(eng) };
    //         h_inputData[index( i, SIZE, j )] = complex_type { temp, -temp };
    //     }
    // }

    printf("Running cufftMalloc\n");
    cufftMalloc<cufftComplex, SIZE, BATCH>( cufftHostData, signalSize, fftPlan );

    printf("Running cufftdxMalloc\n");
    cufftdxMalloc<ARCH, cufftComplex, SIZE, BATCH, FPB, EPT>( cufftDxHostData );

    // Verify cuFFT and cuFFTDx have the same results
    printf( "Compare cuFFT and cuFFTDx (cudaMalloc)\n" );
    // verifyResults<cufftComplex, SIZE, BATCH>( cufftHostData, cufftDxHostData, signalSize );

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
    case 700:
        benchmark<700, 16384, 4096, 1, 32>();
        break;
    case 750:
        benchmark<750, 4096, 4096, 1, 32>();
        break;
    case 800:
        benchmark<800, 32768, 4096, 1, 32>();
        break;
    default:
        printf( "GPU architecture must be 7.0 or greater to use cuFFTDx\n "
                "Skipping Test!\n" );
        break;
    }



    CUDA_RT_CALL( cudaDeviceReset( ) );
}
