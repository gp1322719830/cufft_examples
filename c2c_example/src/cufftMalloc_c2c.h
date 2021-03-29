#include <cufft.h>
#include <cufftXt.h>

#include "../../common/cuda_helper.h"

// Define variables to point at callbacks
#ifdef USE_DOUBLE
__device__ cufftCallbackLoadZ d_loadCallbackPtr   = CB_MulAndScaleInput;
__device__ cufftCallbackStoreZ d_storeCallbackPtr = CB_MulAndScaleOutput;
#else
__device__ cufftCallbackLoadC d_loadCallbackPtr   = CB_MulAndScaleInput;
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_MulAndScaleOutput;
#endif

// cuFFT example using explicit memory copies
template<typename T, typename R, uint SIZE, uint BATCH>
void cufftMalloc( const int &   device,
                  const T *     inputSignal,
                  const T *     multData,
                  const R &     scalar,
                  const size_t &signalSize,
                  fft_params &  fftPlan,
                  T *           outputData ) {

    Timer timer;

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create device data arrays
    T *d_bufferData;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), signalSize ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputSignal, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multData, signalSize, device, NULL ) );

    // Create callback parameters
    cb_inParams<T> h_inParams;
    h_inParams.scale      = scalar;
    h_inParams.multiplier = const_cast<T *>( multData );

    // Copy callback parameters to device
    cb_inParams<T> *d_inParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<T> ), cudaMemcpyHostToDevice ) );

    cb_outParams<T> h_outParams;
    h_outParams.scale      = scalar;
    h_outParams.multiplier = const_cast<T *>( multData );

    cb_outParams<T> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<T> ), cudaMemcpyHostToDevice ) );

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

    // Create host callback pointers
#ifdef USE_DOUBLE
    cufftCallbackLoadZ  h_loadCallbackPtr;
    cufftCallbackStoreZ h_storeCallbackPtr;
#else
    cufftCallbackLoadC  h_loadCallbackPtr;
    cufftCallbackStoreC h_storeCallbackPtr;
#endif

    // Copy device pointers to host
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );

    // Set input callback
#ifdef USE_DOUBLE
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX_DOUBLE, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, ( void ** )&d_outParams ) );
#else
    CUDA_RT_CALL(
        cufftXtSetCallback( fft_inverse, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&d_outParams ) );
#endif

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Malloc\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
#ifdef USE_DOUBLE
        CUDA_RT_CALL( cufftExecZ2Z( fft_forward, const_cast<T *>( inputSignal ), d_bufferData, CUFFT_FORWARD ) );
        CUDA_RT_CALL( cufftExecZ2Z( fft_inverse, d_bufferData, outputData, CUFFT_INVERSE ) );
#else
        CUDA_RT_CALL( cufftExecC2C( fft_forward, const_cast<T *>( inputSignal ), d_bufferData, CUFFT_FORWARD ) );
        CUDA_RT_CALL( cufftExecC2C( fft_inverse, d_bufferData, outputData, CUFFT_INVERSE ) );
#endif
    }
    timer.stopAndPrintGPU( kLoops );

    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( d_bufferData ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );
}
