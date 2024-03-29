include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

SRC = srad_kernel.cu

all: srad

clean:
	rm -rf *.o srad gevo.ptx cuda-device-only-kernel.ll

srad: srad_ptx.cu
	$(CC) $(CFLAGS) -O3 $< -o $@ -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda -lm

srad_nvcc: srad.cu
	nvcc -lineinfo -I$(CUDA_DIR)/include -g $< -o srad -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart_static -lcuda

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -g -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx
