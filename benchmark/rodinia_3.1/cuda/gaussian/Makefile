include ../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60

SRC = gaussian.cu
SRC_PTX = gaussian_ptx.cu

EXE = gaussian

release: $(SRC_PTX)
	$(CC) -O3 $< -o $(EXE) $(CFLAGS) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda
	# $(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean:
	rm gaussian a.ptx cuda-device-only-kernel.ll

llvmir-noopt: $(SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o a.ptx

llvmir-opt: $(SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o a.ptx

nvcc: $(SRC)
	nvcc -O3 -arch=sm_60 -Xptxas -dlcm=cg $< -o $(EXE)
