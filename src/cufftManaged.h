#include <cufft.h>
#include <cufftXt.h>

#include "cuda_helper.h"

// Define variables to point at callbacks
#ifdef USE_DOUBLE
__device__ __managed__ cufftCallbackLoadZ d_loadManagedCallbackPtr   = CB_MulAndScaleInput;
__device__ __managed__ cufftCallbackStoreZ d_storeManagedCallbackPtr = CB_MulAndScaleOutput;
#else
__device__ __managed__ cufftCallbackLoadC d_loadManagedCallbackPtr   = CB_MulAndScaleInput;
__device__ __managed__ cufftCallbackStoreC d_storeManagedCallbackPtr = CB_MulAndScaleOutput;
#endif

// cuFFT example using managed memory copies
template<typename T, typename U, uint SIZE, uint BATCH>
void cufftManaged( const U *     inputSignal,
                   const U *     multData,
                   const size_t &signalSize,
                   fft_params &  fftPlan,
                   T *           h_outputData ) {

    int device = -1;
    CUDA_RT_CALL( cudaGetDevice( &device ) );

    Timer timer;

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create data arrays
    T *inputData;
    T *outputData;
    T *bufferData;

    CUDA_RT_CALL( cudaMallocManaged( &inputData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &outputData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &bufferData, signalSize ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Copy input data to managed allocation
    for ( int i = 0; i < BATCH * SIZE; i += 2 ) {
        inputData[i] = T { inputSignal[i], inputSignal[i + 1] };
    }

    // Create callback parameters
    cb_inParams<T> *inParams;

    CUDA_RT_CALL( cudaMallocManaged( &inParams, sizeof( cb_inParams<T> ) ) );
    inParams->scale = kScale;
    CUDA_RT_CALL( cudaMallocManaged( &inParams->multiplier, signalSize ) );

    // Copy input data to managed allocation
    for ( int i = 0; i < BATCH * SIZE; i += 2 ) {
        inParams->multiplier[i] = T { multData[i], multData[i + 1] };
    }

    cb_outParams<T> *outParams;

    CUDA_RT_CALL( cudaMallocManaged( &outParams, sizeof( cb_outParams<T> ) ) );
    outParams->scale = kScale;
    CUDA_RT_CALL( cudaMallocManaged( &outParams->multiplier, signalSize ) );

    // Copy input data to managed allocation
    for ( int i = 0; i < BATCH * SIZE; i += 2 ) {
        outParams->multiplier[i] = T { multData[i], multData[i + 1] };
    }

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( bufferData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( inParams->multiplier, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outParams->multiplier, signalSize, device, NULL ) );

    CUDA_RT_CALL( cufftPlanMany( &fft_forward,
                                 fftPlan.rank,
                                 fftPlan.n,
                                 fftPlan.inembed,
                                 fftPlan.istride,
                                 fftPlan.idist,
                                 fftPlan.onembed,
                                 fftPlan.ostride,
                                 fftPlan.odist,
#ifdef USE_DOUBLE
                                 CUFFT_Z2Z,
#else
                                 CUFFT_C2C,
#endif
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
#ifdef USE_DOUBLE
                                 CUFFT_Z2Z,
#else
                                 CUFFT_C2C,
#endif
                                 fftPlan.batch ) );

    // Set input callback
#ifdef USE_DOUBLE
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&d_loadManagedCallbackPtr, CUFFT_CB_LD_COMPLEX_DOUBLE, ( void ** )&inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&d_storeManagedCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, ( void ** )&outParams ) );
#else
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&d_loadManagedCallbackPtr, CUFFT_CB_LD_COMPLEX, ( void ** )&inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&d_storeManagedCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&outParams ) );
#endif

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Managed\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
#ifdef USE_DOUBLE
        CUDA_RT_CALL( cufftExecZ2Z( fft_forward, inputData, bufferData, CUFFT_FORWARD ) );
        CUDA_RT_CALL( cufftExecZ2Z( fft_inverse, bufferData, outputData, CUFFT_INVERSE ) );
#else
        CUDA_RT_CALL( cufftExecC2C( fft_forward, inputData, bufferData, CUFFT_FORWARD ) );
        CUDA_RT_CALL( cufftExecC2C( fft_inverse, bufferData, outputData, CUFFT_INVERSE ) );
#endif
    }
    timer.stopAndPrintGPU( kLoops );

    CUDA_RT_CALL( cudaMemcpy( h_outputData, outputData, signalSize, cudaMemcpyDeviceToHost ) );
    // CUDA_RT_CALL( cudaDeviceSynchronize( ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( inputData ) );
    CUDA_RT_CALL( cudaFree( outputData ) );
    CUDA_RT_CALL( cudaFree( bufferData ) );
    CUDA_RT_CALL( cudaFree( inParams->multiplier ) );
    CUDA_RT_CALL( cudaFree( inParams ) );
    CUDA_RT_CALL( cudaFree( outParams->multiplier ) );
    CUDA_RT_CALL( cudaFree( outParams ) );
}
