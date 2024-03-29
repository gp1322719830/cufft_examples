#include <cufft.h>
#include <cufftXt.h>

#include "../../common/cuda_helper.h"

// Define variables to point at callbacks
#ifdef USE_DOUBLE
__device__ cufftCallbackLoadD d_loadCallbackPtr   = CB_MulAndScaleInputR;
__device__ cufftCallbackStoreZ d_storeCallbackPtr = CB_MulAndScaleOutput;
#else
__device__ cufftCallbackLoadR d_loadCallbackPtr   = CB_MulAndScaleInputR;
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_MulAndScaleOutput;
#endif

// cuFFT example using explicit memory copies
template<typename T, typename U, typename R, uint SIZE, uint BATCH>
void cufftMalloc( const int &   device,
                  const T *     inputSignal,
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

    // Create device data arrays
    U *d_bufferData;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), bufferSize ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputSignal, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multDataIn, bufferSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multDataOut, signalSize, device, NULL ) );

    // Create callback parameters
    cb_inParams<U> h_inParams;
    h_inParams.scale      = scalar;
    h_inParams.multiplier = const_cast<U *>( multDataIn );

    // Copy callback parameters to device
    cb_inParams<U> *d_inParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<U> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<U> ), cudaMemcpyHostToDevice ) );

    cb_outParams<T> h_outParams;
    h_outParams.scale      = scalar;
    h_outParams.multiplier = const_cast<T *>( multDataOut );

    cb_outParams<T> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<T> ), cudaMemcpyHostToDevice ) );

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

// Create host callback pointers
#ifdef USE_DOUBLE
    cufftCallbackLoadD  h_loadCallbackPtr;
    cufftCallbackStoreZ h_storeCallbackPtr;
#else
    cufftCallbackLoadR  h_loadCallbackPtr;
    cufftCallbackStoreC h_storeCallbackPtr;
#endif

    // Copy device pointers to host
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );

    // Set input callback
#ifdef USE_DOUBLE
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_REAL_DOUBLE, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, ( void ** )&d_outParams ) );
#else
    CUDA_RT_CALL(
        cufftXtSetCallback( fft_forward, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_REAL, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_forward, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&d_outParams ) );
#endif

    // Execute FFT plan
#ifdef USE_DOUBLE
    std::printf( "cufftExecZ2D/D2Z - FFT/IFFT - Malloc\t" );
#else
    std::printf( "cufftExecC2R/R2C - FFT/IFFT - Malloc\t" );
#endif
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
#ifdef USE_DOUBLE
        CUDA_RT_CALL( cufftExecZ2D( fft_inverse, const_cast<T *>( inputSignal ), d_bufferData ) );
        CUDA_RT_CALL( cufftExecD2Z( fft_forward, d_bufferData, outputData ) );

#else
        CUDA_RT_CALL( cufftExecC2R( fft_inverse, const_cast<T *>( inputSignal ), d_bufferData ) );
        CUDA_RT_CALL( cufftExecR2C( fft_forward, d_bufferData, outputData ) );
#endif
    }
    timer.stopAndPrintGPU( kLoops );

    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( d_bufferData ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );
}
