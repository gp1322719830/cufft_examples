#include <random>

#include <cufft.h>
#include <cufftXt.h>

#include "cuda_helper.h"

// Define variables to point at callbacks
__device__ __managed__ cufftCallbackLoadC d_loadManagedCallbackPtr   = CB_MulAndScaleInputC;
__device__ __managed__ cufftCallbackStoreC d_storeManagedCallbackPtr = CB_MulAndScaleOutputC;

// cuFFT example using managed memory copies
template<typename T>
void cufftManaged( T *h_outputData, const size_t &signalSize, fft_params &fftPlan ) {

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
    std::mt19937 eng;
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            float temp { dist(eng) };
            inputData[i * kDataSize + j] = make_cuComplex( temp, -temp );
        }
    }

    POP_RANGE( )

    printFunction<T>( "Printing input data", inputData );

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

    PUSH_RANGE( "cufftExecC2C - FFT/IFFT - Managed", 7 )
    // Execute FFT plan
    CUDA_RT_CALL( cufftExecC2C( fft_forward, inputData, bufferData, CUFFT_FORWARD ) );
	// CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    // POP_RANGE( )

    // Copy data from device to host
    // CUDA_RT_CALL( cudaMemcpy( h_outputData, bufferData, signalSize, cudaMemcpyDeviceToHost ) );
    // printFunction<T>( "Printing buffer data", h_outputData );

	// PUSH_RANGE( "cufftExecC2C - IFFT - Managed", 8 )
    CUDA_RT_CALL( cufftExecC2C( fft_inverse, bufferData, outputData, CUFFT_INVERSE ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    POP_RANGE( )

    printFunction<T>( "Printing output data", outputData );

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
