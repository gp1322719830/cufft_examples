#include <random>

#include <cufft.h>
#include <cufftXt.h>

#include "cuda_helper.h"

// Define variables to point at callbacks
__device__ cufftCallbackLoadC d_loadCallbackPtr   = CB_MulAndScaleInputC;
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_MulAndScaleOutputC;

// cuFFT example using explicit memory copies
template<typename T, uint SIZE, uint BATCH>
void cufftMalloc( void *inputData, T *h_outputData, const size_t &signalSize, fft_params &fftPlan ) {

    PUSH_RANGE( __FUNCTION__, 1 )

    // Create cufftHandle
    cufftHandle fft_forward;
    cufftHandle fft_inverse;

    // Create host data arrays
    // T *h_inputData = new T[signalSize];

    // Create device data arrays
    T *d_inputData;
    T *d_outputData;
    T *d_bufferData;

    PUSH_RANGE( "Prep Input", 2 )
    // Create input data
    // std::mt19937 eng;
    // std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    // for ( int i = 0; i < BATCH; i++ ) {
    //     for ( int j = 0; j < SIZE; j++ ) {
    //         float temp { dist(eng) };
    //         h_inputData[index( i, SIZE, j )] = make_cuComplex( temp, -temp );
    //     }
    // }

    // Create data
    // std::mt19937 eng;
    // std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for ( int i = 0; i < BATCH * SIZE; i+2 ) {
        // for ( int j = 0; j < SIZE; j++ ) {
    //         float temp { dist(eng) };
            h_inputData[i] = complex_type { inputData[i], -inputData[i+1] };
        // }
    }

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), signalSize ) );

    // Copy input data to device
    CUDA_RT_CALL( cudaMemcpy( d_inputData, inputData, signalSize, cudaMemcpyHostToDevice ) );
    POP_RANGE( )

    PUSH_RANGE( "CB Params", 3 )

    // Create multiplier data
    T *h_multiplier = new T[signalSize];

    for ( int i = 0; i < BATCH; i++ ) {
        for ( int j = 0; j < SIZE; j++ ) {
            h_multiplier[index( i, SIZE, j )] = make_cuFloatComplex( kMultiplier, kMultiplier );
        }
    }

    T *d_multiplier;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplier ), signalSize ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplier, h_multiplier, signalSize, cudaMemcpyHostToDevice ) );

    // Create callback parameters
    cb_inParams<T> h_inParams;
    h_inParams.scale      = kScale;
    h_inParams.multiplier = d_multiplier;

    // Copy callback parameters to device
    cb_inParams<T> *d_inParams;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<T> ), cudaMemcpyHostToDevice ) );

    cb_outParams<T> h_outParams;
    h_outParams.scale      = kScale;
    h_outParams.multiplier = d_multiplier;

    cb_outParams<T> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<T> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<T> ), cudaMemcpyHostToDevice ) );

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

    PUSH_RANGE( "CB Pointers", 5 )
    // Create host callback pointers
    cufftCallbackLoadC  h_loadCallbackPtr;
    cufftCallbackStoreC h_storeCallbackPtr;

    // Copy device pointers to host
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
    CUDA_RT_CALL( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );
    POP_RANGE( )

    PUSH_RANGE( "cufftXtSetCallback", 6 )
    // Set input callback
    CUDA_RT_CALL(
        cufftXtSetCallback( fft_inverse, ( void ** )&h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX, ( void ** )&d_inParams ) );

    // Set output callback
    CUDA_RT_CALL( cufftXtSetCallback(
        fft_inverse, ( void ** )&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, ( void ** )&d_outParams ) );
    POP_RANGE( )

    // Execute FFT plan
    for (int i = 0; i < kLoops; i++) {
        PUSH_RANGE( "cufftExecC2C - FFT/IFFT - Malloc", 7 )
        CUDA_RT_CALL( cufftExecC2C( fft_forward, d_inputData, d_bufferData, CUFFT_FORWARD ) );
        CUDA_RT_CALL( cufftExecC2C( fft_inverse, d_bufferData, d_outputData, CUFFT_INVERSE ) );
        CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    }
    POP_RANGE( )

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );

    // Cleanup Memory
    // delete[]( h_inputData );
    delete[]( h_multiplier );
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );
    CUDA_RT_CALL( cudaFree( d_bufferData ) );
    CUDA_RT_CALL( cudaFree( d_multiplier ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );

    POP_RANGE( )
}
