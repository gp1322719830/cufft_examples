#include <iostream>
#include <limits>
#include <stdexcept>  // std::runtime_error
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <nvToolsExt.h>

#include "/home/belt/Downloads/nvidia-cufftdx-0.0.1-Linux/cufftdx/include/cufftdx.hpp"
#include "/home/belt/Downloads/nvidia-cufftdx-0.0.1-Linux/cufftdx/example/block_io.hpp"

#define PRINT 0

// *************** FOR NVTX *******************
const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof( colors ) / sizeof(uint32_t);

#define PUSH_RANGE( name, cid ) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx( &eventAttrib ); \
}
// *************** FOR NVTX *******************

#define POP_RANGE() nvtxRangePop();

constexpr int kDataSize = 1024;
constexpr int kBatch = 4;
constexpr int kRank = 1;
constexpr int kElementsPerThread = 4;
constexpr float kScale = 2.0f;
constexpr float kMultiplier = 4.0f;
constexpr float kTolerance = 0.001; // Used to compare cuFFT / cuFFTDx results

constexpr int index( int i, int j, int k ) {
	return ( i * j + k );
}

template<typename T>
struct cb_inParams {
	T * multiplier;
	float scale;
};

template<typename T>
struct cb_outParams {
	T * multiplier;
	float scale;
};

typedef struct _fft_params {
	int rank;       	// --- 1D FFTs
	int n[kRank];   	// --- Size of the Fourier transform
	int istride;		// --- Distance between two successive input elements
	int ostride;    	// --- Distance between two successive output elements
	int idist;			// --- Distance between input batches
	int odist; 			// --- Distance between output batches
	int inembed[kRank]; // --- Input size with pitch (ignored for 1D transforms)
	int onembed[kRank]; // --- Output size with pitch (ignored for 1D transforms)
	int batch;      	// --- Number of batched executions
} fft_params;

// Complex multiplication
template<typename T>
__device__ T ComplexScale( T const & a, float const & scale ) {
	T c;
	c.x = a.x * scale;
	c.y = a.y * scale;
	return ( c );
}

// Complex multiplication
template<typename T>
__device__ T ComplexMul( T const & a, T const & b ) {
	T c;
	c.x = a.x * b.x;
	c.y = a.y * b.y;
	return ( c );
}

// Input Callback
template<typename T>
__device__ T CB_MulAndScaleInputC( void *dataIn, size_t offset, void *callerInfo, void *sharedPtr ) {
	cb_inParams<T> * params = static_cast<cb_inParams<T>*>( callerInfo );
	return ( ComplexScale( ComplexMul( static_cast<T*>( dataIn )[offset], ( params->multiplier )[offset] ),
			params->scale ) );
}

// Output Callback
template<typename T>
__device__ void CB_MulAndScaleOutputC( void *dataOut, size_t offset, T element, void *callerInfo, void *sharedPtr ) {
	cb_outParams<T> * params = static_cast<cb_outParams<T>*>( callerInfo );
	static_cast<T*>( dataOut )[offset] = ComplexScale( ComplexMul( element, ( params->multiplier )[offset] ),
			params->scale );
}

// Define variables to point at callbacks
__device__ cufftCallbackLoadC d_loadCallbackPtr = CB_MulAndScaleInputC;
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_MulAndScaleOutputC;

// Define variables to point at callbacks
__device__ __managed__ cufftCallbackLoadC d_loadManagedCallbackPtr = CB_MulAndScaleInputC;
__device__ __managed__ cufftCallbackStoreC d_storeManagedCallbackPtr = CB_MulAndScaleOutputC;

// cuFFTDx Forward FFT CUDA kernel
template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(
		typename FFT::value_type* inputData,
		typename FFT::value_type* outputData ) {
	using complex_type = typename FFT::value_type;

	extern __shared__ complex_type shared_mem[];

	// Local array and copy data into it
	complex_type thread_data[FFT::storage_size];

	// ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
	const unsigned int local_fft_id = threadIdx.y;

	// Load data from global memory to registers
	example::io < FFT > ::load( inputData, thread_data, local_fft_id );

	// Execute FFT
	FFT().execute( thread_data, shared_mem );

	// Save results
	example::io < FFT > ::store( thread_data, outputData, local_fft_id );
}

// cuFFTDx Inverse FFT CUDA kernel
template<class IFFT, typename T>
__launch_bounds__(IFFT::max_threads_per_block) __global__ void block_ifft_kernel(
		typename IFFT::value_type *inputData,
		typename IFFT::value_type *outputData,
		cb_inParams<T> *inParams,
		cb_outParams<T> *outParams ) {

	using complex_type = typename IFFT::value_type;

	extern __shared__ complex_type shared_mem[];

	// Local array and copy data into it
	complex_type thread_data[IFFT::storage_size];

	// ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
	const unsigned int local_fft_id = threadIdx.y;

	// Load data from global memory to registers
	example::io < IFFT > ::load( inputData, thread_data, local_fft_id );

	// Execute input callback functionality
	const uint offset = example::io < IFFT > ::batch_offset( local_fft_id );
	const uint stride = example::io < IFFT > ::stride_size();
	uint index = offset + threadIdx.x;
	for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
		thread_data[i] = ComplexScale( ComplexMul( thread_data[i], ( inParams->multiplier )[index] ), inParams->scale );
		index += stride;
	}

	// Execute FFT
	IFFT().execute( thread_data, shared_mem );

	// Execute output callback functionality
	index = offset + threadIdx.x;
	for ( int i = 0; i < IFFT::elements_per_thread; i++ ) {
		thread_data[i] = ComplexScale( ComplexMul( thread_data[i], ( outParams->multiplier )[index] ),
				outParams->scale );
		index += stride;
	}

	// Save results
	example::io < IFFT > ::store( thread_data, outputData, local_fft_id );
}

// Helper function to print
template<typename T>
void printFunction( T * data ) {
	for ( int i = 0; i < kBatch; i++ )
		for ( int j = 0; j < kDataSize; j++ )
			printf( "Re = %0.2f; Im = %0.2f\n", data[index( i, kDataSize, j )].x, data[index( i, kDataSize, j )].y );
}

template<typename T>
void verifyResults( T const *cufftHostData, T const*cufftDxHostData ) {

	T const * a = cufftHostData;
	T const * b = cufftDxHostData;

	for ( int i = 0; i < kBatch; i++ )
		for ( int j = 0; j < kDataSize; j++ ) {
			if ( std::fabs( a[index( i, kDataSize, j )].x - b[index( i, kDataSize, j )].x ) > kTolerance )
				printf( "R - Batch %d: Element %d: %f - %f (%f) > %f\n", i, j, a[index( i, kDataSize, j )].x,
						b[index( i, kDataSize, j )].x,
						std::fabs( a[index( i, kDataSize, j )].x - b[index( i, kDataSize, j )].x ), kTolerance );
			if ( std::fabs( a[index( i, kDataSize, j )].y - b[index( i, kDataSize, j )].y ) > kTolerance )
				printf( "I - Batch %d: Element %d: %f - %f (%f) > %f\n", i, j, a[index( i, kDataSize, j )].y,
						b[index( i, kDataSize, j )].y,
						std::fabs( a[index( i, kDataSize, j )].y - b[index( i, kDataSize, j )].y ), kTolerance );
		}

//	printf( "All values match\n" );
}

// Warm-up function identical to cufftMalloc
//void warmUpFunction( const int & signalSize, fft_params & fftPlan ) {
//
//	// Create cufftHandle
//	cufftHandle handle;
//
//	// Create host data arrays
//	cufftComplex *h_inputData = new cufftComplex[signalSize];
//	cufftComplex *h_outputData = new cufftComplex[signalSize];
//
//	for ( int i = 0; i < kBatch; i++ )
//		for ( int j = 0; j < kDataSize; j++ )
//			h_inputData[i * kDataSize + j] = make_cuComplex( ( i + j ), ( i - j ) );
//
//	// Create device data arrays
//	cufftComplex *d_inputData;
//	cufftComplex *d_outputData;
//
//	checkCudaErrors( cudaMalloc( (void** )&d_inputData, signalSize ) );
//	checkCudaErrors( cudaMalloc( (void** )&d_outputData, signalSize ) );
//
//	// Copy input data to device
//	checkCudaErrors( cudaMemcpy( d_inputData, h_inputData, signalSize, cudaMemcpyHostToDevice ) );
//
//	// Create callback parameters
//	cb_inParams h_params;
//	h_params.scale = 2.0f;
//
//	// Copy callback parameters to device
//	cb_inParams *d_params;
//	checkCudaErrors( cudaMalloc( (void ** )&d_params, sizeof(cb_inParams) ) );
//	checkCudaErrors( cudaMemcpy( d_params, &h_params, sizeof(cb_inParams), cudaMemcpyHostToDevice ) );
//
//	checkCudaErrors(
//			cufftPlanMany( &handle, fftPlan.rank, fftPlan.n, fftPlan.inembed, fftPlan.istride, fftPlan.idist,
//					fftPlan.onembed, fftPlan.ostride, fftPlan.odist, CUFFT_C2C, fftPlan.batch ) );
//
//	// Create host callback pointers
//	cufftCallbackLoadC h_loadCallbackPtr;
//	cufftCallbackStoreC h_storeCallbackPtr;
//
//	// Copy device pointers to host
//	checkCudaErrors( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
//	checkCudaErrors( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );
//
//	// Set input callback
//	checkCudaErrors(
//			cufftXtSetCallback( handle, (void ** ) &h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX, (void ** )&d_params ) );
//
//	// Set output callback
//	checkCudaErrors(
//			cufftXtSetCallback( handle, (void ** ) &h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, (void ** )&d_params ) );
//
//	// Execute FFT plan
//	checkCudaErrors( cufftExecC2C( handle, d_inputData, d_outputData, CUFFT_FORWARD ) );
//
//	// Cleanup Memory
//	free( h_inputData );
//	free( h_outputData );
//	checkCudaErrors( cudaFree( d_inputData ) );
//	checkCudaErrors( cudaFree( d_outputData ) );
//}

// cuFFT example using explicit memory copies
template<typename T>
void cufftMalloc( T * h_outputData, const int & signalSize, fft_params & fftPlan ) {

	PUSH_RANGE( __FUNCTION__, 1 )

	// Create cufftHandle
	cufftHandle fft_forward;
	cufftHandle fft_inverse;

	// Create host data arrays
	cufftComplex *h_inputData = new cufftComplex[signalSize];

	// Create device data arrays
	cufftComplex *d_inputData;
	cufftComplex *d_outputData;
	cufftComplex *d_bufferData;

	PUSH_RANGE( "Prep Input", 2 )
	for ( int i = 0; i < kBatch; i++ )
		for ( int j = 0; j < kDataSize; j++ )
			h_inputData[index( i, kDataSize, j )] = make_cuComplex( ( i + j ), ( i - j ) );

#if PRINT
	printf( "\nPrinting input data\n" );
	printFunction( h_inputData );
#endif

	checkCudaErrors( cudaMalloc( (void** )&d_inputData, signalSize ) );
	checkCudaErrors( cudaMalloc( (void** )&d_outputData, signalSize ) );
	checkCudaErrors( cudaMalloc( (void** )&d_bufferData, signalSize ) );

	// Copy input data to device
	checkCudaErrors( cudaMemcpy( d_inputData, h_inputData, signalSize, cudaMemcpyHostToDevice ) );
	POP_RANGE()

	PUSH_RANGE( "CB Params", 3 )
	// Create callback parameters
	cufftComplex *h_multiplier = new cufftComplex[signalSize];
	for ( int i = 0; i < kBatch; i++ )
		for ( int j = 0; j < kDataSize; j++ )
			h_multiplier[index( i, kDataSize, j )] = make_cuComplex( kMultiplier, kMultiplier );

	cufftComplex *d_multiplier;
	checkCudaErrors( cudaMalloc( (void** )&d_multiplier, signalSize ) );
	checkCudaErrors( cudaMemcpy( d_multiplier, h_multiplier, signalSize, cudaMemcpyHostToDevice ) );

	cb_inParams<cufftComplex> h_inParams;
	h_inParams.scale = kScale;
	h_inParams.multiplier = d_multiplier;

	// Copy callback parameters to device
	cb_inParams<cufftComplex> *d_inParams;
	checkCudaErrors( cudaMalloc( (void ** )&d_inParams, sizeof(cb_inParams<cufftComplex> ) ) );
	checkCudaErrors(
			cudaMemcpy( d_inParams, &h_inParams, sizeof(cb_inParams<cufftComplex> ), cudaMemcpyHostToDevice ) );

	cb_outParams<cufftComplex> h_outParams;
	h_outParams.scale = kScale;
	h_outParams.multiplier = d_multiplier;

	cb_outParams<cufftComplex> *d_outParams;
	checkCudaErrors( cudaMalloc( (void ** )&d_outParams, sizeof(cb_outParams<cufftComplex> ) ) );
	checkCudaErrors(
			cudaMemcpy( d_outParams, &h_outParams, sizeof(cb_outParams<cufftComplex> ), cudaMemcpyHostToDevice ) );

	POP_RANGE()

	PUSH_RANGE( "cufftPlanMany", 4 )
	checkCudaErrors(
			cufftPlanMany( &fft_forward, fftPlan.rank, fftPlan.n, fftPlan.inembed, fftPlan.istride, fftPlan.idist,
					fftPlan.onembed, fftPlan.ostride, fftPlan.odist, CUFFT_C2C, fftPlan.batch ) );
	checkCudaErrors(
			cufftPlanMany( &fft_inverse, fftPlan.rank, fftPlan.n, fftPlan.inembed, fftPlan.istride, fftPlan.idist,
					fftPlan.onembed, fftPlan.ostride, fftPlan.odist, CUFFT_C2C, fftPlan.batch ) );
	POP_RANGE()

	PUSH_RANGE( "CB Pointers", 5 )
	// Create host callback pointers
	cufftCallbackLoadC h_loadCallbackPtr;
	cufftCallbackStoreC h_storeCallbackPtr;

	// Copy device pointers to host
	checkCudaErrors( cudaMemcpyFromSymbol( &h_loadCallbackPtr, d_loadCallbackPtr, sizeof( h_loadCallbackPtr ) ) );
	checkCudaErrors( cudaMemcpyFromSymbol( &h_storeCallbackPtr, d_storeCallbackPtr, sizeof( h_storeCallbackPtr ) ) );
	POP_RANGE()

	PUSH_RANGE( "cufftXtSetCallback", 6 )
	// Set input callback
	checkCudaErrors(
			cufftXtSetCallback( fft_inverse, (void ** ) &h_loadCallbackPtr, CUFFT_CB_LD_COMPLEX,
					(void ** )&d_inParams ) );

	// Set output callback
	checkCudaErrors(
			cufftXtSetCallback( fft_inverse, (void ** ) &h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX,
					(void ** )&d_outParams ) );
	POP_RANGE()

	PUSH_RANGE( "cufftExecC2C", 7 )
	// Execute FFT plan
	checkCudaErrors( cufftExecC2C( fft_forward, d_inputData, d_bufferData, CUFFT_FORWARD ) );

#if PRINT
	checkCudaErrors( cudaDeviceSynchronize() );
	// Copy data from device to host
	checkCudaErrors( cudaMemcpy( h_outputData, d_bufferData, signalSize, cudaMemcpyDeviceToHost ) );
	printf( "\nPrinting buffer data\n" );
	printFunction( h_outputData );
#endif

	checkCudaErrors( cufftExecC2C( fft_inverse, d_bufferData, d_outputData, CUFFT_INVERSE ) );

	checkCudaErrors( cudaDeviceSynchronize() );
	POP_RANGE()

	// Copy data from device to host
	checkCudaErrors( cudaMemcpy( h_outputData, d_outputData, signalSize, cudaMemcpyDeviceToHost ) );

#if PRINT
	printf( "\nPrinting output data\n" );
	printFunction( h_outputData );
#endif

	// Cleanup Memory
	free( h_inputData );
	free( h_multiplier );
	checkCudaErrors( cudaFree( d_inputData ) );
	checkCudaErrors( cudaFree( d_outputData ) );
	checkCudaErrors( cudaFree( d_multiplier ) );

	POP_RANGE()
}

template<uint A, typename T>
void cuFFTDxMalloc( T * h_outputData ) {

	PUSH_RANGE( __FUNCTION__, 1 )

	// FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
	// will be executed on block level. Shared memory is required for co-operation between threads.
	using FFT = decltype(cufftdx::Block() + cufftdx::Size<kDataSize>() + cufftdx::Type<cufftdx::fft_type::c2c>()
			+ cufftdx::Direction<cufftdx::fft_direction::forward>() + cufftdx::Precision<float>()
			+ cufftdx::ElementsPerThread<kElementsPerThread>() + cufftdx::FFTsPerBlock<kBatch>() + cufftdx::SM<A>());

	using IFFT = decltype(cufftdx::Block() + cufftdx::Size<kDataSize>() + cufftdx::Type<cufftdx::fft_type::c2c>()
			+ cufftdx::Direction<cufftdx::fft_direction::inverse>() + cufftdx::Precision<float>()
			+ cufftdx::ElementsPerThread<kElementsPerThread>() + cufftdx::FFTsPerBlock<kBatch>() + cufftdx::SM<A>());

	using complex_type = typename FFT::value_type;

	// Allocate managed memory for input/output
	auto size = FFT::ffts_per_block * cufftdx::size_of < FFT > ::value;	// cufftdx::Size<5>() * cufftdx::FFTsPerBlock<1>()
	auto sizeBytes = size * sizeof(complex_type);	// Should be same as signalSize

	complex_type *h_inputData = new complex_type[sizeBytes];

	// Create data
	for ( int i = 0; i < kBatch; i++ )
		for ( int j = 0; j < kDataSize; j++ )
			h_inputData[index( i, kDataSize, j )] = complex_type { float( i + j ), float( i - j ) };

#if PRINT
	printf( "\nPrinting input data\n" );
	printFunction( h_inputData );
#endif

	// Create data arrays and allocate
	complex_type* d_inputData;
	complex_type* d_outputData;
	complex_type* d_bufferData;

	checkCudaErrors( cudaMalloc( (void** )&d_inputData, sizeBytes ) );
	checkCudaErrors( cudaMalloc( (void** )&d_outputData, sizeBytes ) );
	checkCudaErrors( cudaMalloc( (void** )&d_bufferData, sizeBytes ) );

	// Copy input data to device
	checkCudaErrors( cudaMemcpy( d_inputData, h_inputData, sizeBytes, cudaMemcpyHostToDevice ) );

	// Create callback parameters
	complex_type *h_multiplier = new complex_type[sizeBytes];
	for ( int i = 0; i < kBatch; i++ )
		for ( int j = 0; j < kDataSize; j++ )
			h_multiplier[index( i, kDataSize, j )] = complex_type { kMultiplier, kMultiplier };

	complex_type *d_multiplier;
	checkCudaErrors( cudaMalloc( (void** )&d_multiplier, sizeBytes ) );
	checkCudaErrors( cudaMemcpy( d_multiplier, h_multiplier, sizeBytes, cudaMemcpyHostToDevice ) );

	cb_inParams<complex_type> h_inParams;
	h_inParams.scale = kScale;
	h_inParams.multiplier = d_multiplier;

	// Copy callback parameters to device
	cb_inParams<complex_type> *d_inParams;
	checkCudaErrors( cudaMalloc( (void ** )&d_inParams, sizeof(cb_inParams<complex_type> ) ) );
	checkCudaErrors(
			cudaMemcpy( d_inParams, &h_inParams, sizeof(cb_inParams<complex_type> ), cudaMemcpyHostToDevice ) );

	cb_outParams<complex_type> h_outParams;
	h_outParams.scale = kScale;
	h_outParams.multiplier = d_multiplier;

	cb_outParams<complex_type> *d_outParams;
	checkCudaErrors( cudaMalloc( (void ** )&d_outParams, sizeof(cb_outParams<complex_type> ) ) );
	checkCudaErrors(
			cudaMemcpy( d_outParams, &h_outParams, sizeof(cb_outParams<complex_type> ), cudaMemcpyHostToDevice ) );

	// Invokes kernel with FFT::block_dim threads in CUDA block
	block_fft_kernel<FFT> <<<1, FFT::block_dim, FFT::shared_memory_size>>>(d_inputData, d_bufferData);
	checkCudaErrors( cudaPeekAtLastError() );

#if PRINT
	// Copy data from device to host
	checkCudaErrors( cudaDeviceSynchronize() );
	checkCudaErrors( cudaMemcpy( h_outputData, d_bufferData, sizeBytes, cudaMemcpyDeviceToHost ) );
	printf( "\nPrinting buffer data\n" );
	printFunction( h_outputData );
#endif

	block_ifft_kernel<IFFT, complex_type> <<<1, FFT::block_dim, FFT::shared_memory_size>>>(d_bufferData, d_outputData, d_inParams, d_outParams);
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors( cudaDeviceSynchronize() );

	// Copy data from device to host
	checkCudaErrors( cudaMemcpy( h_outputData, d_outputData, sizeBytes, cudaMemcpyDeviceToHost ) );

#if PRINT
	printf( "\nPrinting output data\n" );
	printFunction( h_outputData );
#endif

	// Cleanup Memory
	free( h_inputData );
	checkCudaErrors( cudaFree( d_inputData ) );
	checkCudaErrors( cudaFree( d_outputData ) );

	POP_RANGE()
}

//// cuFFT example using managed memory copies
//void useCudaManaged( const int & signalSize, fft_params & fftPlan ) {
//
//	int device = -1;
//	checkCudaErrors( cudaGetDevice( &device ) );
//
//	PUSH_RANGE( __FUNCTION__, 1 )
//
//	// Create cufftHandle
//	cufftHandle handle;
//
//	// Create data arrays
//	cufftComplex *inputData;
//	cufftComplex *outputData;
//
//	PUSH_RANGE( "Prep Input", 2 )
//	cudaMallocManaged( &inputData, signalSize );
//	cudaMallocManaged( &outputData, signalSize );
//
//	checkCudaErrors( cudaMemPrefetchAsync(inputData, signalSize, cudaCpuDeviceId, 0) );
//
//	for ( int i = 0; i < kBatch; i++ )
//		for ( int j = 0; j < kDataSize; j++ )
//			inputData[i * kDataSize + j] = make_cuComplex( ( i + j ), ( i - j ) );
//
////	checkCudaErrors( cudaMemPrefetchAsync( inputData, signalSize, device, 0 ) );
////	checkCudaErrors( cudaMemPrefetchAsync( outputData, signalSize, device, 0 ) );
//	POP_RANGE()
//
//#if PRINT
//	printf( "\nPrinting input data\n" );
//	printFunction<cufftComplex>( inputData );
//#endif
//
//	PUSH_RANGE( "CB Params", 3 )
//	// Create callback parameters
//	cb_inParams *params;
//	cudaMallocManaged( &params, sizeof(cb_inParams) );
//	params->scale = kScale;
////	checkCudaErrors( cudaMemPrefetchAsync( params, sizeof(cb_params), device, 0 ) );
//	POP_RANGE()
//
//	PUSH_RANGE( "cufftPlanMany", 4 )
//	checkCudaErrors(
//			cufftPlanMany( &handle, fftPlan.rank, fftPlan.n, fftPlan.inembed, fftPlan.istride, fftPlan.idist,
//					fftPlan.onembed, fftPlan.ostride, fftPlan.odist, CUFFT_C2C, fftPlan.batch ) );
//	POP_RANGE()
//
//	PUSH_RANGE( "cufftXtSetCallback", 6 )
//	// Set input callback
//	checkCudaErrors(
//			cufftXtSetCallback( handle, (void ** ) &d_loadManagedCallbackPtr, CUFFT_CB_LD_COMPLEX,
//					(void ** )&params ) );
//
//	// Set output callback
//	checkCudaErrors(
//			cufftXtSetCallback( handle, (void ** ) &d_storeManagedCallbackPtr, CUFFT_CB_ST_COMPLEX,
//					(void ** )&params ) );
//	POP_RANGE()
//
//	PUSH_RANGE( "cufftExecC2C", 7 )
//	// Execute FFT plan
//	checkCudaErrors( cufftExecC2C( handle, inputData, outputData, CUFFT_FORWARD ) );
//
//	checkCudaErrors( cudaDeviceSynchronize() );
//	POP_RANGE()
//
//#if PRINT
//	printf( "\nPrinting output data\n" );
//	printFunction<cufftComplex>( outputData );
//#endif
//
//	// Cleanup Memory
//	checkCudaErrors( cudaFree( inputData ) );
//	checkCudaErrors( cudaFree( outputData ) );
//
//	POP_RANGE()
//}

// Returns CUDA device compute capability

uint get_cuda_device_arch( ) {
	int device;
	checkCudaErrors( cudaGetDevice( &device ) );

	cudaDeviceProp props;
	checkCudaErrors( cudaGetDeviceProperties( &props, device ) );

	return static_cast<uint>( props.major ) * 100 + static_cast<unsigned>( props.minor ) * 10;
}

int main( int argc, char **argv ) {

	// Calculate size of signal array to process
	size_t signalSize = sizeof(cufftComplex) * kDataSize * kBatch;

	// Set fft plan parameters
	fft_params fftPlan = { kRank, { kDataSize }, 1, 1, kDataSize, kDataSize, { 0 }, { 0 }, kBatch };

	// Retrieve GPU architecture
	const uint arch = get_cuda_device_arch();

	cuFloatComplex *cufftHostData = new cuFloatComplex[signalSize];
	cuFloatComplex *cufftDxHostData = new cuFloatComplex[signalSize];

//	warmUpFunction( signalSize, fftPlan );

	cufftMalloc < cuFloatComplex > ( cufftHostData, signalSize, fftPlan );

#if PRINT
	printf( "\nPrinting cufftHostData data\n" );
	printFunction( cufftHostData );
#endif

//	useCudaManaged( signalSize, fftPlan );

	switch ( arch ) {
	case 700:
		cuFFTDxMalloc<700, cuFloatComplex>( cufftDxHostData );
#if PRINT
		printf( "\nPrinting cufftDxHostData data\n" );
		printFunction( cufftDxHostData );
#endif
		break;
	case 750:
		cuFFTDxMalloc<750, cuFloatComplex>( cufftDxHostData );
#if PRINT
		printf( "\nPrinting cufftDxHostData data\n" );
		printFunction( cufftDxHostData );
#endif
		break;
	default:
		printf( "GPU architecture must be 7.0 or greater to use cuFFTDx\n Skipping Test!\n" );
		break;
	}

	// Verify cuFFT and cuFFTDx have the same results
	verifyResults( cufftHostData, cufftDxHostData );
}
