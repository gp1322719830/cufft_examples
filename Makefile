NVCC	:=nvcc -ccbin g++
CFLAGS	:=-O3 -std=c++17
ARCHES	:=-gencode arch=compute_75,code=\"compute_75,sm_75\"
INC_DIR	:=-I/home/belt/workStuff/cufft/libcufftdx/include
LIB_DIR	:=
LIBS	:=-lcufft_static -lculibos
SRCDIR	:=./src
OBJDIR	:=obj

ifneq ($(origin PRINT), undefined)
	ifneq ($(PRINT), 0)
		CFLAGS += -DPRINT
	endif
endif

ifneq ($(origin USE_NVTX), undefined)
	ifneq ($(USE_NVTX), 0)
		CFLAGS += -DUSE_NVTX
	endif
endif

SOURCES := cuFFT_vs_cuFFTDx

OBJECTS=$(addprefix $(OBJDIR)/, $(SOURCES:%=%.o))

all: build cuFFT_vs_cuFFTDx
.PHONY: all

build:	
	@mkdir -p $(OBJDIR)

cuFFT_vs_cuFFTDx: $(OBJDIR)/cuFFT_vs_cuFFTDx.o
	$(NVCC) --cudart=static $(CFLAGS) ${ARCHES} $^ -o $@ $(LIBS)

$(OBJDIR)/cuFFT_vs_cuFFTDx.o: $(SRCDIR)/cuFFT_vs_cuFFTDx.cu
	$(NVCC) -dc $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $< -o $@ $(LIBS)
	
clean:
	@echo 'Cleaning up...'
	@echo 'rm -rf $(SOURCES) $(OBJDIR)/*.o'
	@rm -rf $(SOURCES) $(OBJDIR)/*.o
