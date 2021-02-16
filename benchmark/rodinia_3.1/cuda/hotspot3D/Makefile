include ../../common/make.config

CC := clang++

INCLUDE := $(CUDA_DIR)/include
CFLAGS := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_30

SRC = 3D.cu

EXE = 3D

OUTPUT = *.out

FLAGS = #-g -G #-arch sm_20 --ptxas-options=-v
release: $(SRC)
	$(CC) -O3 $(CFLAGS) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT)

llvmir-noopt: opt1.cu
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: opt1.cu
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc cuda-device-only-kernel.ll -o gevo.ptx
