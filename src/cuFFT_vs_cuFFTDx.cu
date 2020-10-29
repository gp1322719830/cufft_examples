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

    int                        N = 1 << 20;
    thrust::device_vector<int> d_x( N, 2 );  // alloc and copy host to device
    thrust::device_vector<int> d_y( N, 4 );

    // Perform SAXPY on 1M elements
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
    printf("Running cufftMalloc\n");
    cufftMalloc<cufftComplex>( cufftHostData, signalSize, fftPlan );

    printFunction( "Printing cufftHostData data", cufftHostData );

    printf("Running cufftManaged\n");
    cufftManaged<cufftComplex>( cufftManagedHostData, signalSize, fftPlan );

    printFunction( "Printing cufftManagedHostData data", cufftManagedHostData );

    // Verify cuFFT (cudaMalloc vs cudaManagedMalloc) have the same results
    printf( "Compare cuFFT (cudaMalloc vs cudaMallocManaged)\n" );
    verifyResults( cufftHostData, cufftManagedHostData, signalSize );

    // Retrieve GPU architecture
    const uint arch { get_cuda_device_arch( ) };

    // Run cuFFTDx example to replicate cuFFT functionality
    printf("Running cufftdxMalloc\n");
    switch ( arch ) {
    case 750:
        cufftdxMalloc<750, cufftComplex>( cufftDxHostData );
        printFunction( "Printing cufftDxHostData data", cufftDxHostData );
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
