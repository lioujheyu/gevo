#makefile

include ../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

SRC = ex_particle_CUDA_float_seq.cu
SRC_PTX = ex_particle_CUDA_float_seq_ptx.cu

particlefilter_float: $(SRC_PTX)
	$(CC) $(CFLAGS) -O3 $< -o $@ -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda -lm
	#$(CC) $(CFLAGS) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda -g -lm -O3 -use_fast_math $< -o $@

clean:
	rm -f particlefilter_naive particlefilter_float cuda-device-only-kernel.ll gevo.ptx

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

nvcc: $(SRC)
	nvcc -arch=sm_60 -Xptxas -dlcm=cg $< -o particlefilter_float
