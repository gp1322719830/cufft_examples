# Getting Started
These examples utilize the following toolsets:
* cuFFT
* cuFFTDx (Requires joining CUDA Math Library Early Access Program) https://developer.nvidia.com/CUDAMathLibraryEA
* Thrust
* C++11
* NVIDIA Tools Extension (NVTX)

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

### Notes
1. This code utilizes cuFFT Callbacks
- https://devblogs.nvidia.com/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
2. This code utilizes NVTX for visual profiling
- https://devblogs.nvidia.com/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
3. This code utilizes separate compilation and linking
- https://devblogs.nvidia.com/separate-compilation-linking-cuda-device-code/
