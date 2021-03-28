#include <cufft.h>
#include <cufftXt.h>

#include "../../common/cuda_helper.h"

// Define variables to point at callbacks
#ifdef USE_DOUBLE
__device__ __managed__ cufftCallbackLoadD d_loadManagedCallbackPtr   = CB_MulAndScaleInputR;
__device__ __managed__ cufftCallbackStoreZ d_storeManagedCallbackPtr = CB_MulAndScaleOutput;
#else
__device__ __managed__ cufftCallbackLoadR d_loadManagedCallbackPtr   = CB_MulAndScaleInputR;
__device__ __managed__ cufftCallbackStoreC d_storeManagedCallbackPtr = CB_MulAndScaleOutput;
#endif

// cuFFT example using managed memory copies
template<typename T, typename U, typename R, uint SIZE, uint BATCH>
void cufftManaged( const T *     inputSignal,
                   const U *     multDataIn,
                   const T *     multDataOut,
                   const R &     scalar,
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
    U *bufferData;

    CUDA_RT_CALL( cudaMallocManaged( &inputData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &outputData, signalSize ) );
    CUDA_RT_CALL( cudaMallocManaged( &bufferData, signalSize / 2 ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Create callback parameters
    cb_inParams<U> *inParams;

    // Only need half size signalSize -- Complex 2 Real
    CUDA_RT_CALL( cudaMallocManaged( &inParams, sizeof( cb_inParams<U> ) ) );
    CUDA_RT_CALL( cudaMallocManaged( &inParams->multiplier, signalSize / 2 ) );
    inParams->scale = scalar;

    cb_outParams<T> *outParams;

    CUDA_RT_CALL( cudaMallocManaged( &outParams, sizeof( cb_outParams<T> ) ) );
    CUDA_RT_CALL( cudaMallocManaged( &outParams->multiplier, signalSize ) );
    outParams->scale = scalar;

    for ( int i = 0; i < BATCH * SIZE; i++ ) {
        inputData[i]             = inputSignal[i];
        inParams->multiplier[i]  = multDataIn[i];
        outParams->multiplier[i] = multDataOut[i];
    }

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( bufferData, signalSize / 2, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( inParams->multiplier, signalSize / 2, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outParams->multiplier, signalSize, device, NULL ) );

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
                                 CUFFT_Z2D,
#else
                                 CUFFT_C2R,
#endif
                                 fftPlan.batch ) );
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
                                 CUFFT_D2Z,
#else
                                 CUFFT_R2C,
#endif
                                 fftPlan.batch ) );

    // Set input callback
#ifdef USE_DOUBLE
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&d_loadManagedCallbackPtr, CUFFT_CB_LD_REAL_DOUBLE, ( void ** )&inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&d_storeManagedCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, ( void ** )&outParams ) );
#else
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&d_loadManagedCallbackPtr, CUFFT_CB_LD_REAL, ( void ** )&inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&d_storeManagedCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&outParams ) );
#endif

    // Execute FFT plan
#ifdef USE_DOUBLE
    std::printf( "cufftExecZ2D/D2Z - FFT/IFFT - Managed\t" );
#else
    std::printf( "cufftExecC2R/R2C - FFT/IFFT - Managed\t" );
#endif
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
#ifdef USE_DOUBLE
        CUDA_RT_CALL( cufftExecZ2D( fft_inverse, inputData, bufferData ) );
        CUDA_RT_CALL( cufftExecD2Z( fft_forward, bufferData, outputData ) );
#else
        CUDA_RT_CALL( cufftExecC2R( fft_inverse, inputData, bufferData ) );
        CUDA_RT_CALL( cufftExecR2C( fft_forward, bufferData, outputData ) );
#endif
    }
    timer.stopAndPrintGPU( kLoops );

    CUDA_RT_CALL( cudaMemcpy( h_outputData, outputData, signalSize, cudaMemcpyDeviceToHost ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( inputData ) );
    CUDA_RT_CALL( cudaFree( outputData ) );
    CUDA_RT_CALL( cudaFree( bufferData ) );
    CUDA_RT_CALL( cudaFree( inParams->multiplier ) );
    CUDA_RT_CALL( cudaFree( inParams ) );
    CUDA_RT_CALL( cudaFree( outParams->multiplier ) );
    CUDA_RT_CALL( cudaFree( outParams ) );
}
