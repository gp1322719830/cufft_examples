
#ifndef CUFFTDX_EXAMPLE_BLOCK_IO_HPP_
#define CUFFTDX_EXAMPLE_BLOCK_IO_HPP_

#include "fp16_common.hpp"

namespace example {
    namespace __io {
        template<bool InRRIILayout = false>
        inline __device__ cufftdx::complex<__half2> convert_to_rrii(const cufftdx::complex<__half2>& value) {
            return to_rrii(value);
        }
        template<>
        inline __device__ cufftdx::complex<__half2> convert_to_rrii<true>(const cufftdx::complex<__half2>& value) {
            return value;
        }
        template<bool InRIRILayout = false>
        inline __device__ cufftdx::complex<__half2> convert_to_riri(const cufftdx::complex<__half2>& value) {
            return to_riri(value);
        }
        template<>
        inline __device__ cufftdx::complex<__half2> convert_to_riri<true>(const cufftdx::complex<__half2>& value) {
            return value;
        }
    } // namespace __io

    template<class FFT, typename T>
    struct io {
        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static inline __device__ unsigned int stride_size() {
            return cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        }

        static inline __device__ unsigned int batch_offset(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return cufftdx::size_of<FFT>::value * global_fft_id;
        }

        // input - global input with all FFTs
        // thread_data - local thread array to load values from input to
        // local_fft_id - ID of FFT batch in CUDA block
        static inline __device__ void load(const T* input,
                                           T*       thread_data,
                                           unsigned int        local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                thread_data[i] = input[index];
                index += stride;
            }
        }

        // If InputInRRIILayout is false, then function assumes that values in input are in RIRI
        // layout, and before loading them to thread_data they are converted to RRII layout.
        // Otherwise, if InputInRRIILayout is true, then function assumes values in input are in RIRI
        // layout, and don't need to be converted before loading to thread_data.
        template<bool InputInRRIILayout = false, class ComplexType = complex_type>
        static inline __device__ void load(const cufftdx::complex<__half2>* input,
                                           cufftdx::complex<__half2>*       thread_data,
                                           unsigned int                     local_fft_id) {
            static_assert(std::is_same<ComplexType, cufftdx::complex<__half2>>::value,
                          "This can be only used with half precision FFTs");
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                thread_data[i] = __io::convert_to_rrii<InputInRRIILayout>(input[index]);
                index += stride;
            }
        }

        static inline __device__ void store(const T* thread_data,
                                            T*       output,
                                            unsigned int        local_fft_id) {
            const unsigned int offset = batch_offset(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index] = thread_data[i];
                index += stride;
            }
        }

        // Function assumes that values in thread_data are in RRII layout.
        // If OutputInRRIILayout is false, values are saved into output in RIRI layout; otherwise - in RRII.
        template<bool OutputInRRIILayout = false, class ComplexType = complex_type>
        static inline __device__ void store(const cufftdx::complex<__half2>* thread_data,
                                            cufftdx::complex<__half2>*       output,
                                            unsigned int                     local_fft_id) {
            static_assert(std::is_same<ComplexType, cufftdx::complex<__half2>>::value,
                          "This can be only used with half precision FFTs");
            const unsigned int offset = batch_offset(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index] = __io::convert_to_riri<OutputInRRIILayout>(thread_data[i]);
                index += stride;
            }
        }

        static inline __device__ unsigned int batch_offset_r2c(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return ((cufftdx::size_of<FFT>::value / 2) + 1) * global_fft_id;
        }

        static inline __device__ void load_r2c(const scalar_type* input,
                                               complex_type*      thread_data,
                                               unsigned int       local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                reinterpret_cast<scalar_type*>(thread_data)[i] = input[index];
                index += stride;
            }
        }

        static inline __device__ void store_r2c(const complex_type* thread_data,
                                                complex_type*       output,
                                                unsigned int        local_fft_id) {
            const unsigned int offset = batch_offset_r2c(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) {
                output[index] = thread_data[i];
                index += stride;
            }
            constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
            constexpr unsigned int values_left_to_store =
                threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
            if (threadIdx.x < values_left_to_store) {
                output[index] = thread_data[FFT::elements_per_thread / 2];
            }
        }

        // Function assumes that values in thread_data are in RRII layout.
        // If OutputInRRIILayout is false, values are saved into output in RIRI layout; otherwise - in RRII.
        template<bool OutputInRRIILayout = false, class ComplexType = complex_type>
        static inline __device__ void store_r2c(const cufftdx::complex<__half2>* thread_data,
                                                cufftdx::complex<__half2>*       output,
                                                unsigned int                     local_fft_id) {
            const unsigned int offset = batch_offset_r2c(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) {
                output[index] = __io::convert_to_riri<OutputInRRIILayout>(thread_data[i]);
                index += stride;
            }
            constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
            constexpr unsigned int values_left_to_store =
                threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
            if (threadIdx.x < values_left_to_store) {
                output[index] = __io::convert_to_riri<OutputInRRIILayout>(thread_data[FFT::elements_per_thread / 2]);
            }
        }

        static inline __device__ unsigned int batch_offset_c2r(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return ((cufftdx::size_of<FFT>::value / 2) + 1) * global_fft_id;
        }

        static inline __device__ void load_c2r(const complex_type* input,
                                               complex_type*       thread_data,
                                               unsigned int        local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset_c2r(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) {
                thread_data[i] = input[index];
                index += stride;
            }
            constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
            constexpr unsigned int values_left_to_load =
                threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
            if (threadIdx.x < values_left_to_load) {
                thread_data[FFT::elements_per_thread / 2] = input[index];
            }
        }

        // If InputInRRIILayout is false, then function assumes that values in input are in RIRI
        // layout, and before loading them to thread_data they are converted to RRII layout.
        // Otherwise, if InputInRRIILayout is true, then function assumes values in input are in RIRI
        // layout, and don't need to be converted before loading to thread_data.
        template<bool InputInRRIILayout = false, class ComplexType = complex_type>
        static inline __device__ void load_c2r(const cufftdx::complex<__half2>* input,
                                               cufftdx::complex<__half2>*       thread_data,
                                               unsigned int                     local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset_c2r(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) {
                thread_data[i] = __io::convert_to_rrii<InputInRRIILayout>(input[index]);
                index += stride;
            }
            constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
            constexpr unsigned int values_left_to_load =
                threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
            if (threadIdx.x < values_left_to_load) {
                thread_data[FFT::elements_per_thread / 2] = __io::convert_to_rrii<InputInRRIILayout>(input[index]);
            }
        }

        static inline __device__ void store_c2r(const complex_type* thread_data,
                                                scalar_type*        output,
                                                unsigned int        local_fft_id) {
            const unsigned int offset = batch_offset(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
				// printf("%f :: %d: %d: %d: || %d: %d: %d: %d || %d\n", threadIdx.x, threadIdx.y, blockIdx.x, i, offset, stride, index);
                output[index] = reinterpret_cast<const scalar_type*>(thread_data)[i];
				// printf("O: %f :: %d: %d: %d: || %d: %d: %d: %d\n", output[index], threadIdx.x, threadIdx.y, blockIdx.x, i, offset, stride, index);
                index += stride;
            }
        }
    };

    template<class FFT>
    struct io_fp16 {
        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static_assert(std::is_same<scalar_type, __half2>::value, "This IO class is only for half precision FFTs");

        static inline __device__ unsigned int stride_size() {
            return cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
        }

        static inline __device__ unsigned int batch_offset(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return cufftdx::size_of<FFT>::value * global_fft_id;
        }

        static inline __device__ void load(const __half2* input, complex_type* thread_data, unsigned int local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = stride_size();
            unsigned int       index        = offset + threadIdx.x;
            const unsigned int batch_stride = FFT::ffts_per_block * cufftdx::size_of<FFT>::value * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                thread_data[i] = to_rrii(input[index], input[index + batch_stride]);
                index += stride;
            }
        }

        static inline __device__ void store(const complex_type* thread_data,
                                            __half2*            output,
                                            unsigned int        local_fft_id) {
            const unsigned int offset       = batch_offset(local_fft_id);
            const unsigned int stride       = stride_size();
            unsigned int       index        = offset + threadIdx.x;
            const unsigned int batch_stride = FFT::ffts_per_block * cufftdx::size_of<FFT>::value * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index]                = to_ri1(thread_data[i]);
                output[index + batch_stride] = to_ri2(thread_data[i]);
                index += stride;
            }
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_BLOCK_IO_HPP_
