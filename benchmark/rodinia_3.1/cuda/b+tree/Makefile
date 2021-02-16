include ../../common/make.config

CUDA_LIB = -lcuda -lcudart
CUDA_FLAG = --cuda-gpu-arch=sm_60
# CUDA_FLAG = -arch sm_35

CC = clang
CUDAC = clang
SRC = main.c kernel/kernel_gpu_cuda_wrapper_ptx.cu
UTIL_SRC = util/timer/timer.c util/num/num.c util/cuda/cuda.cu

INCLUDE=-Iutil/timer/ -I

b+tree.out: $(SRC) ./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o
	$(CUDAC) -O3 $(CUDA_FLAG) $^ -o $@ -L$(CUDA_LIB_DIR) -lm -ldl -lrt -pthread -lcudart_static -lcuda

nvcc: main.c kernel/kernel_gpu_cuda_wrapper.cu ./util/timer/timer.o ./util/num/num.o ./util/cuda/cuda.o
	nvcc -O3 -Xptxas -dlcm=cg -arch=sm_60 $^ -o b+tree.out

./util/timer/timer.o: ./util/timer/timer.c ./common.h ./util/timer/timer.h
	$(CC) -c -O3 $< -o $@

./util/num/num.o: ./util/num/num.c ./common.h ./util/num/num.h
	$(CC) -c -O3 $< -o $@

./util/cuda/cuda.o:	./util/cuda/cuda.cu ./common.h ./util/cuda/cuda.h
	$(CUDAC) $(CUDA_FLAG) -c -O3 $< -o $@

clean:
	find ./ -name "*.o" -type f -delete
	rm -f b+tree.out
	rm -f output.txt

llvmir-noopt: kernel/kernel_gpu_cuda_wrapper.cu
	$(CC) $(CUDA_FLAG) -O0 --cuda-device-only -S -emit-llvm $^ -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

llvmir-opt: kernel/kernel_gpu_cuda_wrapper.cu
	$(CC) $(CUDA_FLAG) -O3 --cuda-device-only -S -emit-llvm $^ -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx60 cuda-device-only-kernel.ll -o gevo.ptx

# b+tree.out:	./main.o \
# 		./kernel/kernel_gpu_cuda_wrapper.o \
# 		./kernel/kernel_gpu_cuda_wrapper_2.o \
# 		./util/timer/timer.o \
# 		./util/num/num.o \
# 		./util/cuda/cuda.o
# 	$(C_C)	./main.o \
# 			./kernel/kernel_gpu_cuda_wrapper.o \
# 			./kernel/kernel_gpu_cuda_wrapper_2.o \
# 			./util/timer/timer.o \
# 			./util/num/num.o \
# 			./util/cuda/cuda.o \
# 			-lm \
# 			-L$(CUDA_LIB_DIR) \
# 			$(CUDA_LIB)\
#                        -o b+tree.out

# main.o:	./common.h \
# 		./main.h \
# 		./main.c
# 	$(C_C)	$(KERNEL_DIM) ./main.c \
# 			-c \
# 			-o ./main.o \
# 			-O3
# ./kernel/kernel_gpu_cuda_wrapper.o:	./common.h \
# 									./kernel/kernel_gpu_cuda_wrapper.h \
# 									./kernel/kernel_gpu_cuda_wrapper.cu
# 	$(CUD_C) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper.cu \
# 				-c \
# 				-o ./kernel/kernel_gpu_cuda_wrapper.o \
# 				-O3 \
# 				$(CUDA_FLAG)

# ./kernel/kernel_gpu_cuda_wrapper_2.o:	./common.h \
# 										./kernel/kernel_gpu_cuda_wrapper_2.h \
# 										./kernel/kernel_gpu_cuda_wrapper_2.cu
# 	$(CUD_C) $(KERNEL_DIM) ./kernel/kernel_gpu_cuda_wrapper_2.cu \
# 				-c \
# 				-o ./kernel/kernel_gpu_cuda_wrapper_2.o \
# 				-O3 \
# 				$(CUDA_FLAG)

