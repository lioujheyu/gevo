DNA_SRC = dna_kernel.cu
AA_SRC = aa_kernel.cu
CC = clang++
CFLAGS = --cuda-gpu-arch=sm_60 -I./

dna-llvmir-opt: $(DNA_SRC)
	$(CC) $(CFLAGS) -O3 -g1 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx

dna-llvmir: $(DNA_SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx

aa-llvmir-opt: $(AA_SRC)
	$(CC) $(CFLAGS) -O3 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx

aa-llvmir: $(AA_SRC)
	$(CC) $(CFLAGS) -O0 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx

