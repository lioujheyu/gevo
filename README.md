# CUDA_EVOLVE

## Package prerequisite

### NVIDIA CUDA

```bash
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt install cuda-9.2
```

### LLVM

```bash
sudo bash -c 'echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" > /etc/apt/sources.list.d/llvm.list'
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo apt update
sudo apt install llvm-8 llvm-dev-8 clang-8
```

### Python and the related packages

```bash
sudo apt install python3 python3-pip
pip3 install deap --user
pip3 install matplotlib --user
pip3 install python-pptx
```

## Installation