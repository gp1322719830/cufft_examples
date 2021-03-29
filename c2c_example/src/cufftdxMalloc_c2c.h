#include <cufftdx.hpp>

#include <cub/block/block_load.cuh>

#include "../../common/cuda_helper.h"

// cuFFTDx Forward FFT && Inverse FFT CUDA kernel
// Parameter T could be used throughout
template<class FFT, class IFFT, typename T>
__launch_bounds__( IFFT::max_threads_per_block ) __global__
    void block_fft_ifft_kernel( const T *              inputData,
                                T *                    outputData,
                                const cb_inParams<T> * inParams,
                                const cb_outParams<T> *outParams ) {

    using complex_type = typename FFT::value_type;
    using scalar_type  = typename complex_type::value_type;

    typedef cub::BlockLoad<complex_type, FFT::block_dim.x, FFT::storage_size, cub::BLOCK_LOAD_STRIPED>   BlockLoad;
    typedef cub::BlockStore<complex_type, FFT::block_dim.x, FFT::storage_size, cub::BLOCK_STORE_STRIPED> BlockStore;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size];
    complex_type temp_mult[FFT::storage_size];

    scalar_type temp_scale {};

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    unsigned int global_fft_id =
        FFT::ffts_per_block == 1 ? blockIdx.x : ( blockIdx.x * FFT::ffts_per_block + threadIdx.y );

    global_fft_id *= cufftdx::size_of<FFT>::value;

    BlockLoad( ).Load( reinterpret_cast<const complex_type *>( inputData ) + global_fft_id, thread_data );
    temp_scale = inParams->scale;

    // Execute FFT
    FFT( ).execute( thread_data, shared_mem );

    BlockLoad( ).Load( reinterpret_cast<const complex_type *>( inParams->multiplier ) + global_fft_id, temp_mult );

#pragma unroll FFT::elements_per_thread
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexMul( thread_data[i], temp_mult[i] );
        thread_data[i] = ComplexScale( thread_data[i], temp_scale );
    }

    // Execute FFT
    IFFT( ).execute( thread_data, shared_mem );

    BlockLoad( ).Load( reinterpret_cast<const complex_type *>( outParams->multiplier ) + global_fft_id, temp_mult );
    temp_scale = outParams->scale;

#pragma unroll FFT::elements_per_thread
    for ( int i = 0; i < FFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexMul( thread_data[i], temp_mult[i] );
        thread_data[i] = ComplexScale( thread_data[i], temp_scale );
    }

    // Save results
    BlockStore( ).Store( reinterpret_cast<complex_type *>( outputData ) + global_fft_id, thread_data );
}

template<typename T, typename R, uint A, uint SIZE, uint BATCH, uint FPB, uint EPT>
void cufftdxMalloc( const int &   device,
                    const T *     inputSignal,
                    const T *     multData,
                    const R &     scalar,
                    const size_t &signalSize,
                    T *           outputData ) {

    Timer timer;

    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    using FFT_base = decltype( cufftdx::Block( ) + cufftdx::Size<SIZE>( ) + cufftdx::Precision<R>( ) +
                               cufftdx::Type<cufftdx::fft_type::c2c>( ) + cufftdx::ElementsPerThread<EPT>( ) +
                               cufftdx::FFTsPerBlock<FPB>( ) + cufftdx::SM<A>( ) );

    using FFT  = decltype( FFT_base( ) + cufftdx::Direction<cufftdx::fft_direction::forward>( ) );
    using IFFT = decltype( FFT_base( ) + cufftdx::Direction<cufftdx::fft_direction::inverse>( ) );

    // Increase dynamic memory limit if required.
    CUDA_RT_CALL( cudaFuncSetAttribute(
        block_fft_ifft_kernel<FFT, IFFT, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size ) );

    // Create data arrays and allocate
    CUDA_RT_CALL( cudaMemPrefetchAsync( inputSignal, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( multData, signalSize, device, NULL ) );

    // Create callback parameters
    cb_inParams<T> *inParams;
    CUDA_RT_CALL( cudaMallocManaged( &inParams, sizeof( cb_inParams<T> ) ) );
    inParams->scale      = scalar;
    inParams->multiplier = const_cast<T *>( multData );

    cb_outParams<T> *outParams;
    CUDA_RT_CALL( cudaMallocManaged( &outParams, sizeof( cb_outParams<T> ) ) );
    outParams->scale      = scalar;
    outParams->multiplier = const_cast<T *>( multData );

    CUDA_RT_CALL( cudaMemPrefetchAsync( inParams, sizeof( cb_inParams<T> ), device, NULL ) );
    CUDA_RT_CALL( cudaMemPrefetchAsync( outParams, sizeof( cb_outParams<T> ), device, NULL ) );

    unsigned int blocks_per_grid { static_cast<unsigned int>( std::ceil( BATCH / FPB ) ) };

    // Execute FFT plan
    std::printf( "cufftExecC2C - FFT/IFFT - Dx\t\t" );
    timer.startGPUTimer( );

    for ( int i = 0; i < kLoops; i++ ) {
        block_fft_ifft_kernel<FFT, IFFT, T><<<blocks_per_grid, FFT::block_dim, FFT::shared_memory_size>>>(
            const_cast<T *>( inputSignal ), outputData, inParams, outParams );
    }
    timer.stopAndPrintGPU( kLoops );

    CUDA_RT_CALL( cudaMemPrefetchAsync( outputData, signalSize, cudaCpuDeviceId, 0 ) );

    // Cleanup Memory
    CUDA_RT_CALL( cudaFree( inParams ) );
    CUDA_RT_CALL( cudaFree( outParams ) );
}