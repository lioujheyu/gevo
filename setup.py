#!/usr/bin/env python

import sys
try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

setup(
    name='gevo',
    version='1.2.2',
    python_requires='>=3.6.1',
    description='Optimize CUDA kernel code using Evolutionary Computation',
    author='Jhe-Yu Liou',
    author_email='lioujheyu@gmail.com',
    url='https://github.com/lioujheyu/gevo',
    install_requires=[
        'deap>=1.2',
        'matplotlib>=2.1',
        'python-pptx>=0.6',
        'psutil',
        'rich>=9.11',
        'pandas',
        'pycuda'
    ],
    packages=['gevo', 'fuzzycompare', 'llvm-mutate'],
    scripts=[
        'bin/gevo-evolve',
        'bin/gevo-analyze',
        'bin/gevo-evaluate',
        'bin/gevo-stage-analyze',
        'bin/gevo-explore',
        'bin/gevo-reduce',
        'bin/llvm-mutate'
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Code Generators',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
    ],
    cmake_install_dir="gevo"
)
