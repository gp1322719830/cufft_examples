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
void cufftManaged( const int & device,const T *     inputSignal,
                   const U *     multDataIn,
                   const T *     multDataOut,
                   const R &     scalar,
                   const size_t &signalSize,
                   const size_t &bufferSize,
                   fft_params &  fftPlan,
                   T *           outputData ) {

    Timer timer;

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create data arrays
    U *bufferData;
    CUDA_RT_CALL( cudaMallocManaged( &bufferData, bufferSize ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputSignal, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multDataIn, bufferSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multDataOut, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( bufferData, bufferSize, device, NULL ) );

    // Create callback parameters
    cb_inParams<U> *inParams;

    // Only need half size signalSize -- Complex 2 Real
    CUDA_RT_CALL( cudaMallocManaged( &inParams, sizeof( cb_inParams<U> ) ) );
    inParams->scale = scalar;
    inParams->multiplier = const_cast<U*>(multDataIn);

    cb_outParams<T> *outParams;

    CUDA_RT_CALL( cudaMallocManaged( &outParams, sizeof( cb_outParams<T> ) ) );
    outParams->scale = scalar;
    outParams->multiplier = const_cast<T*>(multDataOut);

    CUDA_RT_CALL( cudaMemPrefetchAsync( inParams, sizeof( cb_inParams<U> ), device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outParams, sizeof( cb_outParams<T> ), device, NULL ) );

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
        CUDA_RT_CALL( cufftExecZ2D( fft_inverse, const_cast<T*>(inputSignal), bufferData ) );
        CUDA_RT_CALL( cufftExecD2Z( fft_forward, bufferData, outputData ) );
#else
        CUDA_RT_CALL( cufftExecC2R( fft_inverse, const_cast<T*>(inputSignal), bufferData ) );
        CUDA_RT_CALL( cufftExecR2C( fft_forward, bufferData, outputData ) );
#endif
    }
    timer.stopAndPrintGPU( kLoops );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( bufferData ) );
}
