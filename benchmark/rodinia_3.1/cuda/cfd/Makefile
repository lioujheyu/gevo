include ../../common/make.config

CC=clang++
NVCC_FLAG := -I$(CUDA_DIR)/include --cuda-gpu-arch=sm_60
CUDA_SDK_PATH := $(SDK_DIR)

# Determine the correct version of the cutil library
CUTIL_LIB = # -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = #-lcutil_x86_64
     endif
endif

#SRC=euler3d.cu
SRC=euler3d_ptx.cu

euler3d: $(SRC)
	$(CC) $(KERNEL_DIM) $(NVCC_FLAG) -O3 $< -o $@ -I$(CUDA_SDK_PATH)/common/inc \
		-L$(CUDA_SDK_PATH) -L$(CUDA_LIB_DIR) $(CUTIL_LIB) -ldl -lrt -pthread -lcudart_static -lcuda

clean:
	rm -f euler3d *.linkinfo

llvmir-noopt: euler3d.cu
	$(CC) $(NVCC_FLAG) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll -I$(CUDA_SDK_PATH)/common/inc
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: euler3d.cu
	$(CC) $(NVCC_FLAG) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll -I$(CUDA_SDK_PATH)/common/inc
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

nvcc: euler3d.cu
	nvcc -Xptxas -dlcm=cg $(KERNEL_DIM) -I$(CUDA_SDK_PATH)/common/inc -I$(CUDA_DIR)/include -arch=sm_60 -O3 $< -o euler3d
