# GEVO

## Package prerequisite
### NVIDIA CUDA

```bash
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt update
sudo apt install cuda-9.2
```

### LLVM

```bash
sudo bash -c 'echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" > /etc/apt/sources.list.d/llvm.list'
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo apt update
sudo apt install llvm-8 llvm-8-dev clang-8
```

The following command only install the meta package which soft-link the current clang/llvm binaries.

```bash
sudo apt install llvm clang
```

If the above meta package is not functional as expected, manually soft-link the necessary binaries as follows

```bash
sudo ln -s /usr/local/bin/clang++-8 /usr/local/bin/clang++
sudo ln -s /usr/local/bin/clang-8 /usr/local/bin/clang
sudo ln -s /usr/local/bin/opt-8 /usr/local/bin/opt
sudo ln -s /usr/local/bin/llvm-dis-8 /usr/local/bin/llvm-dis
```

### Python and the related packages

```bash
sudo apt install python3 python3-pip
pip3 install --user deap
pip3 install --user matplotlib
pip3 install --user python-pptx
pip3 install --user pybind11
```

## Installation
### **llvm-mutate**
GEVO relies on a specific cuda branch of llvm-mutate

```bash
git clone git@github.com:lioujheyu/llvm-mutate.git
cd llvm-mutate
git checkout cuda
mkdir build
cd build
cmake ../
sudo make && make install
```

### Install **GEVO**
```bash
pip3 install gevo
```
