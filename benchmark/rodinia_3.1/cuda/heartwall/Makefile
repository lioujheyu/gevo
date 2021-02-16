include ../../common/make.config

OUTPUT=-DOUTPUT

NVCC_FLAGS=--cuda-gpu-arch=sm_60
NVCC=clang
# NVCC=nvcc
# SRC=main.cu
SRC=main_ptx.cu

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(NVCC) $^ -g -I/usr/local/cuda/include -lm -o $@ -L$(CUDA_LIB_DIR) -ldl -lrt -pthread -lcudart_static -lcuda

# compile main function file into object (binary)
main.o: $(SRC) kernel.cu define.c
	$(NVCC) -g $(NVCC_FLAGS) $(OUTPUT) $(KERNEL_DIM) $< -I./AVI -c -O0 -o main.o

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo

llvmir-noopt: main.cu
	$(NVCC) $(NVCC_FLAGS) -I./AVI -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: main.cu
	$(NVCC) $(NVCC_FLAGS) -I./AVI -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx
