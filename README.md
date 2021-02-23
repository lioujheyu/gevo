

# GEVO

[![PyPI version](https://badge.fury.io/py/gevo.svg)](https://badge.fury.io/py/gevo)

GEVO (Gpu optimization using EVOlutionary computation) is a tool for automatically discovering optimization opportunities and tuning the performance of GPU kernels in the LLVM representation. 

## Package dependency

### Python 3.6.1 or above

Python packages dependency is list in the `setup.py` and `pyproject.toml`, though they will be taken care by python pip.

### NVIDIA CUDA 11

While GEVO can work with any CUDA version so long LLVM compiler can compile it, certain benchmark I test requires Nvidia intrinsic only supported by the most recent CUDA. Thus CUDA 11 is thus enforced.

:exclamation: Note: Since 2018.8, nvidia driver 418 and beyond has a profiling permission restriction. Please refer to [this nvidia document](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
and [this nvidia forum thread](https://forums.developer.nvidia.com/t/nvprof-warning-the-user-does-not-have-permission-to-profile-on-the-target-device/72374/6)
for how to lift the restriction for `nvprof`, which is the profiling tool GEVO use to measure the CUDA kernel runtime and power comsumption.

### LLVM 11

LLVM 11 + CUDA 11.1 has been tested with current GEVO, even though LLVM 11 does not officially support CUDA 11. The oldest LLVM I have used in the past is LLVM 8. It might work, but no guarantee. More information might be found in LLVM document: ["Compileing CUDA with clang"](https://llvm.org/docs/CompileCudaWithLLVM.html) 

Usually, LLVM in the system package management system is much older than the current development. If you are using Ubuntu/Debian, please refer to ["LLVM nightly packages"](https://apt.llvm.org/) for how to install the most recent LLVM into the system. 

## Installation
Simply install GEVO from pypi using pip
```bash
pip3 install --user gevo 
```

Or build manually 
```bash
git clone https://github.com/lioujheyu/gevo
cd gevo
python3 -m build .
pip install dist/gevo-1.2.1-<platform-specific-string>.whl
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

This start the evolution with default parameters. Please use `--help` for more information including the default values for each argument.

## Citation
If this project helps your research, please cite the following paper in your publication.

```
@article{liou2020gevo,
  author = {Liou, Jhe-Yu and Wang, Xiaodong and Forrest, Stephanie and Wu, Carole-Jean},
  title = {GEVO: GPU Code Optimization Using Evolutionary Computation},
  year = {2020},
  journal = {ACM Trans. Archit. Code Optim.}
}
```