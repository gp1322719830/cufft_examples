#include <cufftdx.hpp>

#include "block_io.hpp"
#include "cuda_helper.h"

// cuFFTDx Forward FFT CUDA kernel
template<class FFT>
__launch_bounds__( FFT::max_threads_per_block ) __global__
    void block_fft_kernel( typename FFT::value_type *inputData, typename FFT::value_type *outputData ) {

    using complex_type = typename FFT::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id { threadIdx.y };

    // Load data from global memory to registers
    example::io<FFT>::load( inputData, thread_data, local_fft_id );

    // Execute FFT
    FFT( ).execute( thread_data, shared_mem );

    // Save results
    example::io<FFT>::store( thread_data, outputData, local_fft_id );
}

// cuFFTDx Inverse FFT CUDA kernel
template<class IFFT, typename T>
__launch_bounds__( IFFT::max_threads_per_block ) __global__
    void block_ifft_kernel( typename IFFT::value_type *inputData,
                            typename IFFT::value_type *outputData,
                            cb_inParams<T> *           inParams,
                            cb_outParams<T> *          outParams ) {

    using complex_type = typename IFFT::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array and copy data into it
    complex_type thread_data[IFFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id { threadIdx.y };

    // Load data from global memory to registers
    example::io<IFFT>::load( inputData, thread_data, local_fft_id );

    // Execute input callback functionality
    const uint offset = example::io<IFFT>::batch_offset( local_fft_id );
    const uint stride = example::io<IFFT>::stride_size( );
    uint       index  = offset + threadIdx.x;
    for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
        thread_data[i] = ComplexScale( ComplexMul( thread_data[i], ( inParams->multiplier )[index] ), inParams->scale );
        index += stride;
    }

    // Execute FFT
    IFFT( ).execute( thread_data, shared_mem );

    // Execute output callback functionality
    index = offset + threadIdx.x;

    for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
        thread_data[i] =
            ComplexScale( ComplexMul( thread_data[i], ( outParams->multiplier )[index] ), outParams->scale );
        index += stride;
    }

    // Save results
    example::io<IFFT>::store( thread_data, outputData, local_fft_id );
}

template<uint A, typename T>
void cufftdxMalloc( T *h_outputData ) {

    PUSH_RANGE( __FUNCTION__, 1 )

    PUSH_RANGE( "Prep Input", 2 )

    // FFT is defined, its: size, type, direction, precision. Block() operator
    // informs that FFT will be executed on block level. Shared memory is
    // required for co-operation between threads.
    using FFT = decltype( cufftdx::Block( ) + cufftdx::Size<kDataSize>( ) + cufftdx::Type<cufftdx::fft_type::c2c>( ) +
                          cufftdx::Direction<cufftdx::fft_direction::forward>( ) + cufftdx::Precision<float>( ) +
                          cufftdx::ElementsPerThread<kElementsPerThread>( ) + cufftdx::FFTsPerBlock<kBatch>( ) +
                          cufftdx::SM<A>( ) );

    using IFFT = decltype( cufftdx::Block( ) + cufftdx::Size<kDataSize>( ) + cufftdx::Type<cufftdx::fft_type::c2c>( ) +
                           cufftdx::Direction<cufftdx::fft_direction::inverse>( ) + cufftdx::Precision<float>( ) +
                           cufftdx::ElementsPerThread<kElementsPerThread>( ) + cufftdx::FFTsPerBlock<kBatch>( ) +
                           cufftdx::SM<A>( ) );

    using complex_type = typename FFT::value_type;

    // Allocate managed memory for input/output
    auto size { FFT::ffts_per_block * cufftdx::size_of<FFT>::value };  // cufftdx::Size<5>() *
                                                                       // cufftdx::FFTsPerBlock<1>()
    auto sizeBytes { size * sizeof( complex_type ) };                  // Should be same as signalSize

    complex_type *h_inputData = new complex_type[sizeBytes];

    // Create data
    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_inputData[index( i, kDataSize, j )] = complex_type { float( i + j ), float( i - j ) };
        }
    }

    printFunction( "Printing input data", h_inputData );

    // Create data arrays and allocate
    complex_type *d_inputData;
    complex_type *d_outputData;
    complex_type *d_bufferData;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inputData ), sizeBytes ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outputData ), sizeBytes ) );
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_bufferData ), sizeBytes ) );

    // Copy input data to device
    CUDA_RT_CALL( cudaMemcpy( d_inputData, h_inputData, sizeBytes, cudaMemcpyHostToDevice ) );

    POP_RANGE( )

    PUSH_RANGE( "CB Params", 3 )

    // Create multiplier data
    complex_type *h_multiplier = new complex_type[sizeBytes];

    for ( int i = 0; i < kBatch; i++ ) {
        for ( int j = 0; j < kDataSize; j++ ) {
            h_multiplier[index( i, kDataSize, j )] = complex_type { kMultiplier, kMultiplier };
        }
    }

    complex_type *d_multiplier;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_multiplier ), sizeBytes ) );
    CUDA_RT_CALL( cudaMemcpy( d_multiplier, h_multiplier, sizeBytes, cudaMemcpyHostToDevice ) );

    // Create callback parameters
    cb_inParams<complex_type> h_inParams;
    h_inParams.scale      = kScale;
    h_inParams.multiplier = d_multiplier;

    // Copy callback parameters to device
    cb_inParams<complex_type> *d_inParams;

    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_inParams ), sizeof( cb_inParams<complex_type> ) ) );
    CUDA_RT_CALL( cudaMemcpy( d_inParams, &h_inParams, sizeof( cb_inParams<complex_type> ), cudaMemcpyHostToDevice ) );

    cb_outParams<complex_type> h_outParams;
    h_outParams.scale      = kScale;
    h_outParams.multiplier = d_multiplier;

    cb_outParams<complex_type> *d_outParams;
    CUDA_RT_CALL( cudaMalloc( reinterpret_cast<void **>( &d_outParams ), sizeof( cb_outParams<complex_type> ) ) );
    CUDA_RT_CALL(
        cudaMemcpy( d_outParams, &h_outParams, sizeof( cb_outParams<complex_type> ), cudaMemcpyHostToDevice ) );

    POP_RANGE( )

    PUSH_RANGE( "cufftExecC2C - FFT", 7 )
    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>( d_inputData, d_bufferData );
    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    POP_RANGE( )

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_bufferData, sizeBytes, cudaMemcpyDeviceToHost ) );
    printFunction( "Printing buffer data", h_outputData );

    PUSH_RANGE( "cufftExecC2C - IFFT", 8 )
    block_ifft_kernel<IFFT, complex_type>
        <<<1, FFT::block_dim, FFT::shared_memory_size>>>( d_bufferData, d_outputData, d_inParams, d_outParams );
    CUDA_RT_CALL( cudaPeekAtLastError( ) );
    CUDA_RT_CALL( cudaDeviceSynchronize( ) );
    POP_RANGE( )

    // Copy data from device to host
    CUDA_RT_CALL( cudaMemcpy( h_outputData, d_outputData, sizeBytes, cudaMemcpyDeviceToHost ) );

    printFunction( "Printing output data", h_outputData );

    // Cleanup Memory
    delete[]( h_inputData );
    delete[]( h_multiplier );
    CUDA_RT_CALL( cudaFree( d_inputData ) );
    CUDA_RT_CALL( cudaFree( d_outputData ) );
    CUDA_RT_CALL( cudaFree( d_bufferData ) );
    CUDA_RT_CALL( cudaFree( d_multiplier ) );
    CUDA_RT_CALL( cudaFree( d_inParams ) );
    CUDA_RT_CALL( cudaFree( d_outParams ) );

    POP_RANGE( )
}