# ADEPT (GPU-BSW)
GPU-BSW or GPU Batch Smith-Waterman is a GPU accelerated implementation of the Smith-Waterman alignment algorithm based on the ADEPT strategy hence also referenced as ADEPT. Implementation details of ADEPT can be found in the publication here: https://rdcu.be/b7fhY. ADEPT uses GPU's two level parallelism to perform multiple sequence alignments in batches while using fine grained parallelism to accelerate each individual alignment.  Overall it provides several time faster performance in comparison to existing SIMD implementations for CPU, a comparative study with existing CPU and GPU methods has been provided in the publication mentioned above. ADEPT performs a complete smith-waterman alignment with affine gap penalities and can align both protein and DNA sequences. 

ADEPT provides a driver function that separates CUDA code from the main application which enables easy use and integeration in existing applications, effectively providing a drop in replacement for CPU libraries. The driver also enables balancing of alignments across all the GPUs available on a system.
       

 
### To Build:


`mkdir build `

`cd build `

`cmake CMAKE_BUILD_TYPE=Release .. `

`make `


### To Execute DNA test run: 

`./program_gpu dna ../test-data/dna-reference.fasta ../test-data/dna-query.fasta ./out_file`

### To Execute Protein test run: <br />

`./program_gpu aa ../test-data/protein-reference.fasta ../test-data/protein-query.fasta ./out_file`

### Contact
If you need help modifying the library to match your specific use-case or for other issues and bug reports please open an issue or reach out at mgawan@lbl.gov


### Citation
*Awan, M.G., Deslippe, J., Buluc, A. et al. ADEPT: a domain independent sequence alignment strategy for gpu architectures. BMC Bioinformatics 21, 406 (2020). https://doi.org/10.1186/s12859-020-03720-1*

### License:
        
**GPU accelerated Smith-Waterman for performing batch alignments (GPU-BSW) Copyright (c) 2019, The
Regents of the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved.**

**If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.**

**NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do
so.**
