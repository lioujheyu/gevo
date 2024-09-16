
cuda_file = CUDA_sort.cu CUDAUtils.cu cuda_utils/cuda_device_utils.cu Backends/CUDABaseBackend.cu Backends/MD_CUDABackend.cu Interactions/CUDABaseInteraction.cu Interactions/CUDADNAInteraction.cu Interactions/CUDALJInteraction.cu Interactions/CUDARNAInteraction.cu Interactions/CUDAPatchyInteraction.cu Interactions/CUDATEPInteraction.cu Interactions/CUDAInteractionFactory.cu Lists/CUDAListFactory.cu Lists/CUDANoList.cu Lists/CUDASimpleVerletList.cu Lists/CUDABinVerletList.cu Thermostats/CUDABaseThermostat.cu Thermostats/CUDANoThermostat.cu Thermostats/CUDAThermostatFactory.cu Thermostats/CUDABrownianThermostat.cu Thermostats/CUDASRDThermostat.cu Thermostats/CUDALangevinThermostat.cu Thermostats/CUDABussiThermostat.cu

CFLAGS = --cuda-gpu-arch=sm_60 -I./ -Icuda_utils -IBackends -IInteractions -ILists -IThermostats --std=c++11

compile: all_cuda.cu
	clang++ $(CFLAGS) -O3 -g1 --cuda-device-only -S -emit-llvm all_cuda.cu -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx

compile_nvcc: all_cuda.cu
	nvcc --ptx -I./ -Icuda_utils -IBackends -IInteractions -ILists -IThermostats --std=c++11 all_cuda.cu

compile_test: all_cuda_mod.cu
	clang++ $(CFLAGS) -O3 -g1 --cuda-device-only -S -emit-llvm $< -o cuda-device-only-kernel.ll
	cat cuda-device-only-kernel.ll | llvm-mutate -n -o cuda-device-only-kernel.ll
	llc -march=nvptx64 -mcpu=sm_60 -mattr=+ptx70 cuda-device-only-kernel.ll -o gevo.ptx
