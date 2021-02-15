# GEVO

GEVO (Gpu optimization using EVOlutionary computation) is a tool for automatically discovering optimization opportunities and tuning the performance of GPU kernels in the LLVM representation. 

## Package dependency

### Python 3.6.1 or above

Python packages dependency is list in the `setup.py`, though they will be taken care by python pip.

### NVIDIA CUDA 11

While GEVO can work with any CUDA version so long LLVM compiler can compile it, certain benchmark I test requires Nvidia intrinsic only supported by the most recent CUDA. Thus CUDA 11 is enforced for now.

:exclamation: Note: Since 2018.8, nvidia driver 418 and beyond has a profiling permission restriction. Please refer to [this nvidia document](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
and [this nvidia forum thread](https://forums.developer.nvidia.com/t/nvprof-warning-the-user-does-not-have-permission-to-profile-on-the-target-device/72374/6)
for how to lift the restriction for `nvprof`, which is the profiling tool GEVO use to measure the CUDA kernel runtime and power comsumption.

### LLVM 11

LLVM 11 + CUDA 11.1 has been tested with current GEVO, even though LLVM 11 does not officially support CUDA 11. The oldest LLVM I have used in the past is LLVM 8. It might work, but no guarantee. More information might be found in LLVM document: ["Compileing CUDA with clang"](https://llvm.org/docs/CompileCudaWithLLVM.html) 

Usually, LLVM in the system package management system is much older than the current development. If you are using Ubuntu/Debian, please refer to ["LLVM nightly packages"](https://apt.llvm.org/) for how to install the most recent LLVM into the system. 

GEVO relies on following llvm commands (`clang`, `clang++`, `opt`, `llvm-dis`) without specifying the version in the end. If there are multiple version of LLVM installed, make sure the default llvm commands are matched with the desired version. Otherwise, you need to manually soft-link or using linux command `alternative` to link the right LLVM commands. like the following using soft-link.

```bash
sudo ln -s /usr/local/bin/clang++-11 /usr/local/bin/clang++
sudo ln -s /usr/local/bin/clang-11 /usr/local/bin/clang
sudo ln -s /usr/local/bin/opt-11 /usr/local/bin/opt
sudo ln -s /usr/local/bin/llvm-dis-11 /usr/local/bin/llvm-dis
```

## Installation
### **llvm-mutate**
llvm-mutate is the tool under my development that GEVO use to manipulate LLVM-IR code.
GEVO relies on a specific `cuda` branch of llvm-mutate to manipulate CUDA kernel represented in LLVM-IR. 

```bash
git clone https://github.com/lioujheyu/llvm-mutate
cd llvm-mutate
git checkout cuda
mkdir build
cd build
cmake ../
sudo make && make install
```

### Install **GEVO**
Simply install GEVO from pypi using pip
```bash
pip3 install --user gevo 
```

Or build manually 
```bash
git clone https://github.com/lioujheyu/gevo
cd gevo
python3 setup.py sdist bdist_wheel
pip install dist/gevo-1.1.1-cp38-cp38-linux_x86_64.whl
```

## Evolve a CUDA program from GEVO
### Prepare the CUDA program
GEVO interacts with target CUDA program with following requirement.
* profile as a json file - to specify how to run the target CUDA program with necessary arguments.
* A modified CUDA program that have its kernel(s) loaded externally from `gevo.ptx` file
* A method to compile the kerel code separatedly through LLVM into `cuda-device-only-kernel.ll`

Please check with `profile.json` and Makefile/CUDA code of each benchmark (`under benchmark/rodinia_3.1/cuda`) as examples. 

The detail steps of preparing these files are under construction. 

### Start the evolution
```
gevo-evolve -P <the profile in json form> 
```

This start the evolution with default parameters (population )