#include <cufft.h>
#include <cufftXt.h>

#include "../../common/cuda_helper.h"

// Define variables to point at callbacks
#ifdef USE_DOUBLE
__device__ cufftCallbackLoadZ d_loadCallbackPtr   = CB_MulAndScaleInput;
__device__ cufftCallbackStoreD d_storeCallbackPtr = CB_MulAndScaleOutputR;
#else
__device__ cufftCallbackLoadC d_loadCallbackPtr   = CB_MulAndScaleInput;
__device__ cufftCallbackStoreR d_storeCallbackPtr = CB_MulAndScaleOutputR;
#endif

// cuFFT example using explicit memory copies
template<typename T, typename U, typename R, uint SIZE, uint BATCH>
void cufftMalloc_r2r( const T *     inputSignal,
                      const U *     multDataIn,
                      const T *     multDataOut,
                      const R &     scalar,
                      const size_t &signalSize,
                      fft_params &  fftPlan,
                      T *           h_outputData ) {

    Timer timer;

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create device data arrays
    T *d_inputData;
    T *d_outputData;
    U *d_bufferData;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), signalSize * 2 ) );

    // Copy input data to device
    CUDA_RT_CALL( cudaMemcpy( d_inputData, inputSignal, signalSize, cudaMemcpyHostToDevice ) );

    U *d_multiplierIn;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplierIn ), signalSize * 2 ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplierIn, multDataIn, signalSize * 2, cudaMemcpyHostToDevice ) );

    // Create callback parameters
    cb_inParams<U> h_inParams;
    h_inParams.scale      = scalar;
    h_inParams.multiplier = d_multiplierIn;

    // Copy callback parameters to device
    cb_inParams<U> *d_inParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<U> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<U> ), cudaMemcpyHostToDevice ) );

    T *d_multiplierOut;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplierOut ), signalSize ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplierOut, multDataOut, signalSize, cudaMemcpyHostToDevice ) );

    cb_outParams<T> h_outParams;
    h_outParams.scale      = scalar;
    h_outParams.multiplier = d_multiplierOut;

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
                                 CUFFT_D2Z,
#else
                                 CUFFT_R2C,
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
                                 CUFFT_Z2D,
#else
                                 CUFFT_C2R,
#endif
                                 fftPlan.batch ) );

// Create host callback pointers
#ifdef USE_DOUBLE
    cufftCallbackLoadZ  h_loadCallbackPtr;
    cufftCallbackStoreD h_storeCallbackPtr;
#else
    cufftCallbackLoadC  h_loadCallbackPtr;
    cufftCallbackStoreR h_storeCallbackPtr;
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
        fft_inverse, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_REAL_DOUBLE, ( void ** )&d_outParams ) );
#else
    CUDA_RT_CALL(
        cufftXtSetCallback( fft_inverse, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL(
        cufftXtSetCallback( fft_inverse, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_REAL, ( void ** )&d_outParams ) );
#endif

    // Execute FFT plan
#ifdef USE_DOUBLE
    std::printf( "cufftExecD2Z/Z2D - FFT/IFFT - Managed\t" );
#else
    std::printf( "cufftExecR2C/C2R - FFT/IFFT - Managed\t" );
#endif
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
#ifdef USE_DOUBLE
        CUDA_RT_CALL( cufftExecD2Z( fft_forward, d_inputData, d_bufferData ) );
        CUDA_RT_CALL( cufftExecZ2D( fft_inverse, d_bufferData, d_outputData ) );

#else
        CUDA_RT_CALL( cufftExecR2C( fft_forward, d_inputData, d_bufferData ) );
        CUDA_RT_CALL( cufftExecC2R( fft_inverse, d_bufferData, d_outputData ) );
#endif
    }
    timer.stopAndPrintGPU( kLoops );

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );
    CUDA_RT_CALL( cudaFree( d_bufferData ) );
    CUDA_RT_CALL( cudaFree( d_multiplierIn ) );
    CUDA_RT_CALL( cudaFree( d_multiplierOut ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );
}
