include ../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

SRC = hotspot.cu
SRC_PTX = hotspot_ptx.cu

EXE = hotspot

hotspot: $(SRC_PTX)
	$(CC) -O3 $(CFLAGS) $(KERNEL_DIM) $< -o $@ -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

nvcc: $(SRC)
	nvcc -O3 -arch=sm_60 -I$(INCLUDE) $< -o hotspot
