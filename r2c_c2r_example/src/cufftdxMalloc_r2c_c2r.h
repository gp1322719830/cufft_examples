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
    complex_type thread_data[FFT::storage_size] {};
    complex_type temp_mult[FFT::storage_size] {};

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

    BlockLoad( ).Load( reinterpret_cast<const complex_type *>( inParams->multiplier ) + global_fft_id, temp_mult );

#pragma unroll IFFT::elements_per_thread
    for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexMul( thread_data[i], temp_mult[i] );
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
    // Option 1
    BlockStore_C2R( ).Store( outputData + global_fft_id, reinterpret_cast<R2C &>( thread_data ) );

    // Option 2
    // BlockStore_C2R( ).Store( outputData + global_fft_id,
    //                          *static_cast<T( * )[FFT::storage_size]>( static_cast<void *>( thread_data ) ) );
}

template<typename T, typename U, typename R, uint A, uint SIZE, uint BATCH, uint FPB, uint EPT>
void cufftdxMalloc( const int &   device,
                    const T *     inputSignal,
                    const U *     multDataIn,
                    const T *     multDataOut,
                    const R &     scalar,
                    const size_t &signalSize,
                    const size_t &bufferSize,
                    T *           outputData ) {

    Timer timer;

    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    using FFT_base = decltype( cufftdx::Block( ) + cufftdx::Size<SIZE>( ) + cufftdx::Precision<R>( ) +
                               cufftdx::ElementsPerThread<EPT>( ) + cufftdx::FFTsPerBlock<FPB>( ) + cufftdx::SM<A>( ) );

    using FFT  = decltype( FFT_base( ) + cufftdx::Type<cufftdx::fft_type::r2c>( ) );
    using IFFT = decltype( FFT_base( ) + cufftdx::Type<cufftdx::fft_type::c2r>( ) );

    const auto shared_memory_size = std::max( FFT::shared_memory_size, IFFT::shared_memory_size );

    // Increase dynamic memory limit if required.
    CUDA_RT_CALL( cudaFuncSetAttribute(
        block_fft_ifft_kernel<FFT, IFFT, T, U>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size ) );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inputSignal, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multDataIn, bufferSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multDataOut, signalSize, device, NULL ) );

    // Create callback parameters
    cb_inParams<U> *inParams;
    CUDA_RT_CALL( cudaMallocManaged( &inParams, sizeof( cb_inParams<U> ) ) );
    inParams->scale      = scalar;
    inParams->multiplier = const_cast<U *>( multDataIn );

    cb_outParams<T> *outParams;
    CUDA_RT_CALL( cudaMallocManaged( &outParams, sizeof( cb_outParams<T> ) ) );
    outParams->scale      = scalar;
    outParams->multiplier = const_cast<T *>( multDataOut );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inParams, sizeof( cb_inParams<U> ), device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outParams, sizeof( cb_outParams<T> ), device, NULL ) );

    unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( BATCH / FPB ) ) };

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Dx\t\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
        block_fft_ifft_kernel<FFT, IFFT, T, U><<<blocks_per_grid, FFT::block_dim, FFT::shared_memory_size>>>(
            const_cast<T *>( inputSignal ), outputData, inParams, outParams );
    }
    timer.stopAndPrintGPU( kLoops );

    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( inParams ) );
    CUDA_RT_CALL( cudaFree( outParams ) );
}