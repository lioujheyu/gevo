include ../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

SRC = bfs.cu
SRC_PTX = bfs_ptx.cu

EXE = bfs

release: $(SRC_PTX)
	$(CC) -O3 $< -o $(EXE) $(CFLAGS) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda

clean: $(SRC)
	rm -f $(EXE) gevo.ptx cuda-device-only-kernel.ll $(EXE).linkinfo result.txt

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

nvcc: $(SRC)
	nvcc -lineinfo -arch=sm_60 -I$(CUDA_DIR)/include -Xptxas -dlcm=cg $< -o $(EXE)
