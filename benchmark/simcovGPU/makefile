gpu:
	nvcc -arch=sm_60 -Iinclude -std=c++11 -lcudart_static -lcuda -o simcov_gpu.out src/options.cpp src/tissue.cu src/main.cpp
gpu_debug:
	nvcc -arch=sm_60 -g -G -Iinclude -lcudart_static -lcuda -std=c++11 -o simcov_gpu.out src/options.cpp src/tissue.cu src/main.cpp

llvmir-opt:
	clang++-11 -std=c++11 --cuda-gpu-arch=sm_60 -Iinclude -O3 -g1 --cuda-device-only -S -emit-llvm src/tissue_ptx.cu -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc-11 -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx

clean:
	rm -f simcov_gpu.out cuda-device-only-kernel.ll gevo.ptx

llvm-test:
	clang++-11 --cuda-gpu-arch=sm_60 -Iinclude -O3 -g1 --cuda-device-only -S -emit-llvm src/tissue_ptx_md.cu -o test.ll
	cat test.ll | llvm-mutate -n -o test.ll
	llc-11 -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 test.ll -o gevo.ptx
