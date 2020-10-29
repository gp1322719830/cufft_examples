#pragma once

template<uint A, typename T>
void cuFFTDxMalloc( T *h_outputData );

#include "cufftdx_example.cu"