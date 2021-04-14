#!/usr/bin/env python

import sys
import os
import codecs

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='gevo',
    version=get_version("gevo/__init__.py"),
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
