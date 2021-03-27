#include <cufftdx.hpp>

#include <cub/block/block_load.cuh>

#include "../../common/cuda_helper.h"

// cuFFTDx Forward FFT && Inverse FFT CUDA kernel
template<class FFT, class IFFT, typename T, typename U>
__launch_bounds__( IFFT::max_threads_per_block ) __global__
    void block_fft_ifft_kernel( const T *              inputData,
                                T *                    outputData,
                                const cb_inParams<U> * inParams,
                                const cb_outParams<T> *outParams ) {

    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    typedef cub::BlockLoad<complex_type, FFT::block_dim.x, FFT::storage_size, cub::BLOCK_LOAD_STRIPED> BlockLoad;
    typedef cub::BlockLoad<T, FFT::block_dim.x, FFT::storage_size, cub::BLOCK_LOAD_STRIPED>            BlockLoad_R2C;
    typedef cub::BlockStore<T, FFT::block_dim.x, FFT::storage_size, cub::BLOCK_STORE_STRIPED>          BlockStore_C2R;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size] {};  // U type doesn't work with FFT.execute()
    complex_type temp_mult_i[FFT::storage_size] {};

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    unsigned int global_fft_id =
        FFT::ffts_per_block == 1 ? blockIdx.x : ( blockIdx.x * FFT::ffts_per_block + threadIdx.y );

    global_fft_id *= cufftdx::size_of<FFT>::value;

    // Option 1
    using R2C = T[FFT::storage_size];
    BlockLoad_R2C( ).Load( inputData + global_fft_id, reinterpret_cast<R2C &>( thread_data ) );

    // Option 2
    // BlockLoad_R2C( ).Load( inputData + global_fft_id,
    // *static_cast<T(*)[FFT::storage_size]>(static_cast<void*>(thread_data)));

    // // Execute FFT
    FFT( ).execute( thread_data, shared_mem );

    BlockLoad( ).Load( reinterpret_cast<const complex_type *>( inParams->multiplier ) + global_fft_id, temp_mult_i );

#pragma unroll IFFT::elements_per_thread
    for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexMul( thread_data[i], temp_mult_i[i] );
        thread_data[i] = ComplexScale( thread_data[i], inParams->scale );
    }

    // Execute FFT
    IFFT( ).execute( thread_data, shared_mem );

    unsigned int index = global_fft_id + threadIdx.x;
#pragma unroll FFT::elements_per_thread
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        reinterpret_cast<scalar_type *>( thread_data )[i] *= ( outParams->multiplier[index] * outParams->scale );
        index += ( cufftdx::size_of<FFT>::value / FFT::elements_per_thread );
    }

    // Save results
    Option 1
    BlockStore_C2R( ).Store( outputData + global_fft_id, reinterpret_cast<R2C &>( thread_data ) );

    // Option 2
    // BlockStore_C2R( ).Store( outputData + global_fft_id,
    //                          *static_cast<T( * )[FFT::storage_size]>( static_cast<void *>( thread_data ) ) );
}

template<typename T, typename U, typename R, uint A, uint SIZE, uint BATCH, uint FPB, uint EPT>
void cufftdxMalloc_r2r( const T *     inputSignal,
                        const U *     multDataIn,
                        const T *     multDataOut,
                        const R &     scalar,
                        const size_t &signalSize,
                        T *           h_outputData ) {

    Timer timer;

    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    using FFT_base = decltype( cufftdx::Block( ) + cufftdx::Size<SIZE>( ) + cufftdx::Precision<R>( ) +
                               cufftdx::ElementsPerThread<EPT>( ) + cufftdx::FFTsPerBlock<FPB>( ) + cufftdx::SM<A>( ) );

    using FFT = decltype( FFT_base( ) + cufftdx::Type<cufftdx::fft_type::r2c>( ) );

    using IFFT = decltype( FFT_base( ) + cufftdx::Type<cufftdx::fft_type::c2r>( ) );

    // using complex_type = typename FFT::value_type;
    // using scalar_type  = typename complex_type::value_type;

    // printf( "%d: %d\n", FFT::shared_memory_size, IFFT::shared_memory_size );

    const auto shared_memory_size = std::max( FFT::shared_memory_size, IFFT::shared_memory_size );

    // Increase dynamic memory limit if required.
    CUDA_RT_CALL( cudaFuncSetAttribute(
        block_fft_ifft_kernel<FFT, IFFT, T, U>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size ) );

    // Create data arrays and allocate
    T *d_inputData;
    T *d_outputData;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), signalSize ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), signalSize ) );

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

    unsigned int blocks_per_grid { BATCH / FPB };
    // printf("%d: %d: %d: %d: %d\n", blocks_per_grid, FFT::block_dim.x, FFT::block_dim.y, FFT::block_dim.z,

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Dx\t\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
        block_fft_ifft_kernel<FFT, IFFT, T, U><<<blocks_per_grid, FFT::block_dim, FFT::shared_memory_size>>>(
            d_inputData, d_outputData, d_inParams, d_outParams );
    }
    timer.stopAndPrintGPU( kLoops );

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );

    T *h_bufferData = new T[signalSize];
    CUDA_RT_CALL( cudaMemcpy( h_bufferData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );
    // for ( int i = 0; i < ( SIZE * BATCH ); i++ ) {
    // printf( "%d: %f\n", i, h_outputData[i] );
    // }

    // Cleanup MemoryZ
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );
    CUDA_RT_CALL( cudaFree( d_multiplierIn ) );
    CUDA_RT_CALL( cudaFree( d_multiplierOut ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );
}