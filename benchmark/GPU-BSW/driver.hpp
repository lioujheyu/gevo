#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "kernel.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>
#include "alignments.hpp"

#define NSTREAMS 2

#define NOW std::chrono::high_resolution_clock::now()



namespace gpu_bsw_driver{

// for storing the alignment results
struct alignment_results{
  short* ref_begin;
  short* query_begin;
  short* ref_end;
  short* query_end;
  short* top_scores;
};

size_t 
get_tot_gpu_mem(int id);

void
kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4], float factor = 1.0);


void
kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap, float factor = 1.0);

void
verificationTest(std::string rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
                 short* g_alBend);
}
#endif
