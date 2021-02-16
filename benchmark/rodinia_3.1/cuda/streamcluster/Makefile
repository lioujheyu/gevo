include ../../common/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

EXEC = sc_gpu

SRC = streamcluster_cuda.cu

all: $(EXEC)

clean:
	rm -f *.o *~ *.txt $(EXEC) *.linkinfo gevo.ptx cuda-device-only-kernel.ll

$(EXEC): streamcluster_cuda_cpu.cpp streamcluster_cuda.o
	$(CC) $(CFLAGS) -O3 $? -o $@ -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda -lm

streamcluster_cuda.o: streamcluster_cuda_ptx.cu
	$(CC) $(CFLAGS) -O3 $^ -o $@ -I$(INCLUDE) -c


llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx
