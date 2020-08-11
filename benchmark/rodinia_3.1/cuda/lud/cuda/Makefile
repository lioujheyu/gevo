include ../../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
INCLUDE_LUD := ../common
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

SRC = lud.cu
SRC_PTX = lud_ptx.cu

EXE = lud_cuda

lud_cuda: lud_ptx.o ../common/common.o
	$(CC) -g $(CFLAGS) $(KERNEL_DIM) $? -o $@ -I$(INCLUDE) -I$(INCLUDE_LUD) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda

lud_ptx.o: lud_ptx.cu
	$(CC) -g $(CFLAGS) $(KERNEL_DIM) $< -c -o $@ -I$(INCLUDE) -I$(INCLUDE_LUD)

../common/common.o: ../common/common.c
	clang -g $(CFLAGS) $(KERNEL_DIM) $< -c -o $@ -I$(INCLUDE) -I$(INCLUDE_LUD)

clean: $(SRC)
	rm -f $(EXE) ../common/common.o *.o

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll -I$(INCLUDE_LUD)
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o a.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll -I$(INCLUDE_LUD)
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o a.ptx

