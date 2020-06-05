NVCC	:=nvcc -ccbin g++
CFLAGS	:=-O3 -std=c++14
ARCHES	:=-gencode arch=compute_75,code=\"compute_75,sm_75\"
INC_DIR	:=-I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include/cufftdx/include
LIB_DIR	:=
LIBS	:=-lcufft_static -lculibos

SOURCES := cuFFT_vs_cuFFTDx \

all: $(SOURCES)
.PHONY: all

cuFFT_vs_cuFFTDx: cuFFT_vs_cuFFTDx.o
	$(NVCC) --cudart=static $(CFLAGS) ${ARCHES} $^ -o $@ $(LIBS)

cuFFT_vs_cuFFTDx.o: cuFFT_vs_cuFFTDx.cu
	$(NVCC) -dc $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $^ -o $@ $(LIBS)
	
clean:
	rm -f $(SOURCES) cuFFT_vs_cuFFTDx.o
