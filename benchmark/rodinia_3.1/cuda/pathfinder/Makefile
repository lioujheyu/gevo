include ../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_35

SRC = pathfinder.cu
SRC_PTX = pathfinder_ptx.cu

EXE = pathfinder

pathfinder: $(SRC_PTX)
	$(CC) -O3 $(CFLAGS) $(KERNEL_DIM) $< -o $@ -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc cuda-device-only-kernel.ll -o a.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc cuda-device-only-kernel.ll -o a.ptx

