NVCC	:=nvcc -ccbin g++
CFLAGS	:=-O3 -std=c++17 -DTHRUST_IGNORE_CUB_VERSION_CHECK
ARCHES	:=-gencode arch=compute_70,code=\"compute_70,sm_70\" \
			-gencode arch=compute_75,code=\"compute_75,sm_75\" \
			-gencode arch=compute_80,code=\"compute_80,sm_80\"
INC_DIR	:=-I/${HOME}/workStuff/cufft/libcufftdx/include -I/${HOME}/workStuff/git_examples/cub
LIB_DIR	:=
LIBS	:=-lcufft_static -lculibos
SRCDIR	:=./src
OBJDIR	:=obj

ifneq ($(origin USE_DOUBLE), undefined)
	ifneq ($(USE_DOUBLE), 0)
		CFLAGS += -DUSE_DOUBLE
	endif
endif

ifneq ($(origin PRINT), undefined)
	ifneq ($(PRINT), 0)
		CFLAGS += -DPRINT
	endif
endif

SOURCES := cufftdx_r2c_c2r_example

OBJECTS=$(addprefix $(OBJDIR)/, $(SOURCES:%=%.o))

all: build cufftdx_r2c_c2r_example
.PHONY: all

build:	
	@mkdir -p $(OBJDIR)

cufftdx_r2c_c2r_example: $(OBJDIR)/cufftdx_r2c_c2r_example.o
	$(NVCC) --cudart=static $(CFLAGS) ${ARCHES} $^ -o $@ $(LIBS)

$(OBJDIR)/cufftdx_r2c_c2r_example.o: $(SRCDIR)/cufftdx_r2c_c2r_example.cu
	$(NVCC) -dc $(CFLAGS) $(INC_DIR) $(LIB_DIR) ${ARCHES} $< -o $@ $(LIBS)
	
clean:
	@echo 'Cleaning up...'
	@echo 'rm -rf $(SOURCES) $(OBJDIR)/*.o'
	@rm -rf $(SOURCES) $(OBJDIR)/*.o
