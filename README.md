# NOTE
At the moment, C2C examples require https://github.com/mnicely/cub.

# Getting Started
These examples utilize the following toolsets:
* cuFFT
* cuFFTDx (Requires joining CUDA Math Library Early Access Program) https://developer.nvidia.com/CUDAMathLibraryEA
* C++11

# Hardware
Volta+

## cuFFT_vs_cuFFTDx

This code runs three scenarios
1. cuFFT using cudaMalloc
2. cuFFT using cudaMallocManaged
3. cuFFTDx using cudaMalloc

### Objectives
1. Compare coding styles between cuFFT, using cudaMalloc and cudaMallocManaged
2. Compare performance between cuFFT, using cudaMalloc and cudaMallocManaged
- This is accomplished using NVTX labeling
3. Compare performance and results between cuFFT and cuFFTDx

### Execution
For float
```bash
make
./cuFFT_vs_cuFFTDx
```

For double
```bash
export USE_DOUBLE=1
make
./cuFFT_vs_cuFFTDx
```

To compare results (cuFFT and cuFFTDx are not expected to be exact)
```bash
export PRINT=1
make
./cuFFT_vs_cuFFTDx
```

### Output
```bash
export PRINT=1
exportUSE_DOUBLE=1
make
./cuFFT_vs_cuFFTDx

FFT Size: 2048 -- Batch: 16384 -- FFT Per Block: 1 -- EPT: 16
cufftExecC2C - FFT/IFFT - Malloc        XX.XX ms
cufftExecC2C - FFT/IFFT - Managed       XX.XX ms

Compare results
All values match!

cufftExecC2C - FFT/IFFT - Dx            XX.XX ms

Compare results
All values match!
```

### Notes
1. This code utilizes cuFFT Callbacks
- https://devblogs.nvidia.com/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
2. This code utilizes separate compilation and linking
- https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/
