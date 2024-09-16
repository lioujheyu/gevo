#include "driver.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

size_t gpu_bsw_driver::get_tot_gpu_mem(int id) {
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, id));
  return prop.totalGlobalMem;
}
void
gpu_bsw_driver::kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4], float factor)
{
    short matchScore = scores[0], misMatchScore = scores[1], startGap = scores[2], extendGap = scores[3];
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;

    initialize_alignments(alignments, totalAlignments); // pinned memory allocation
    auto start = NOW;

    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 5 * sizeof(short);

    #pragma omp parallel
    {
      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }

        if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = floor(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);
        unsigned max_alns_sugg = 20000;
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu;
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail*factor<<"\n";

        int BLOCKS_l = alignmentsPerDevice;
        if(my_cpu_id == deviceCount - 1)
            BLOCKS_l += leftOver_device;
        unsigned leftOvers    = BLOCKS_l % its;
        unsigned stringsPerIt = BLOCKS_l / its;
        gpu_alignments gpu_data(stringsPerIt + leftOvers); // gpu mallocs

        short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
        short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
        short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
        short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
        short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;
        unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
        unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
        cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));

        char *strA_d, *strB_d;
        cudaErrchk(cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char)));
        cudaErrchk(cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char)));

        char* strA;
        cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
        char* strB;
        cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));

        float total_packing = 0;

        unsigned int err;
        CUmodule module;
        CUfunction kernel, kernel2;

        err = cuModuleLoad(&module, "gevo.ptx");
        err = cuModuleGetFunction(&kernel, module, "_ZN7gpu_bsw19sequence_dna_kernelEPcS0_PjS1_PsS2_S2_S2_S2_ssss");
        err = cuModuleGetFunction(&kernel2, module, "_ZN7gpu_bsw20sequence_dna_reverseEPcS0_PjS1_PsS2_S2_S2_S2_ssss");

        cudaProfilerStart();

        auto start2 = NOW;
        for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
        {
            auto packing_start = NOW;
            int                                      blocksLaunched = 0;
            std::vector<std::string>::const_iterator beginAVec;
            std::vector<std::string>::const_iterator endAVec;
            std::vector<std::string>::const_iterator beginBVec;
            std::vector<std::string>::const_iterator endBVec;
            if(perGPUIts == its - 1)
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt + leftOvers;
            }
            else
            {
                beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
                beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
                endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) +  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a copy of strings it needs to align
                blocksLaunched = stringsPerIt;
            }

            std::vector<std::string> sequencesA(beginAVec, endAVec);
            std::vector<std::string> sequencesB(beginBVec, endBVec);
            long unsigned running_sum = 0;
            int sequences_per_stream = (blocksLaunched) / NSTREAMS;
            int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
            long unsigned half_length_A = 0;
            long unsigned half_length_B = 0;

            auto start_cpu = NOW;

            for(int i = 0; i < sequencesA.size(); i++)
            {
                running_sum +=sequencesA[i].size();
                offsetA_h[i] = running_sum;//sequencesA[i].size();
                if(i == sequences_per_stream - 1){
                    half_length_A = running_sum;
                    running_sum = 0;
                  }
            }
            long unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

            running_sum = 0;
            for(int i = 0; i < sequencesB.size(); i++)
            {
                running_sum +=sequencesB[i].size();
                offsetB_h[i] = running_sum; //sequencesB[i].size();
                if(i == sequences_per_stream - 1){
                  half_length_B = running_sum;
                  running_sum = 0;
                }
            }
            long unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

            auto end_cpu = NOW;
            std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

            total_time_cpu += cpu_dur.count();
            long unsigned offsetSumA = 0;
            long unsigned offsetSumB = 0;

            for(int i = 0; i < sequencesA.size(); i++)
            {
                char* seqptrA = strA + offsetSumA;
                memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
                char* seqptrB = strB + offsetSumB;
                memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
                offsetSumA += sequencesA[i].size();
                offsetSumB += sequencesB[i].size();
            }

            auto packing_end = NOW;
            std::chrono::duration<double> packing_dur = packing_end - packing_start;

            total_packing += packing_dur.count();

            asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
            unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
            unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
            unsigned alignmentPad = 4 + (4 - totShmem % 4);
            size_t   ShmemBytes = totShmem + alignmentPad;
            if(ShmemBytes > 48000)
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);
                // cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

            void* args[] = {
              &strA_d, &strB_d,
              &gpu_data.offset_ref_gpu, &gpu_data.offset_query_gpu, &gpu_data.ref_start_gpu,
              &gpu_data.ref_end_gpu, &gpu_data.query_start_gpu, &gpu_data.query_end_gpu, &gpu_data.scores_gpu,
              &matchScore, &misMatchScore, &startGap, &extendGap };
            err = cuLaunchKernel(kernel, sequences_per_stream, 1, 1,
                                 minSize, 1, 1,
                                 ShmemBytes, streams_cuda[0], args, 0);
            // gpu_bsw::sequence_dna_kernel<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
            //     strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
            //     gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            auto *strA_d_2 = strA_d + half_length_A;
            auto *strB_d_2 = strB_d + half_length_B;
            auto *gpu_data_offset_ref_gpu_2 = gpu_data.offset_ref_gpu + sequences_per_stream;
            auto *gpu_data_offset_query_gpu_2 = gpu_data.offset_query_gpu + sequences_per_stream;
            auto *gpu_data_ref_start_gpu_2 = gpu_data.ref_start_gpu + sequences_per_stream;
            auto *gpu_data_ref_end_gpu_2 = gpu_data.ref_end_gpu + sequences_per_stream;
            auto *gpu_data_query_start_gpu_2 = gpu_data.query_start_gpu + sequences_per_stream;
            auto *gpu_data_query_end_gpu_2 = gpu_data.query_end_gpu + sequences_per_stream;
            auto *gpu_data_scores_gpu_2 = gpu_data.scores_gpu + sequences_per_stream;
            void* args_streamB[] = {
              &strA_d_2, &strB_d_2,
              &gpu_data_offset_ref_gpu_2, &gpu_data_offset_query_gpu_2, &gpu_data_ref_start_gpu_2,
              &gpu_data_ref_end_gpu_2, &gpu_data_query_start_gpu_2, &gpu_data_query_end_gpu_2, &gpu_data_scores_gpu_2,
              &matchScore, &misMatchScore, &startGap, &extendGap };
            err = cuLaunchKernel(kernel, sequences_per_stream + sequences_stream_leftover, 1, 1,
                                 minSize, 1, 1,
                                 ShmemBytes, streams_cuda[1], args_streamB, 0);
            // gpu_bsw::sequence_dna_kernel<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
            //     strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
            //      gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
            //      gpu_data.scores_gpu + sequences_per_stream, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            // copyin back end index so that we can find new min
            asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);

            cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

            auto sec_cpu_start = NOW;
            int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
            auto sec_cpu_end = NOW;
            std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
            total_time_cpu += dur_sec_cpu.count();

            err = cuLaunchKernel(kernel2, sequences_per_stream, 1, 1,
                                 newMin, 1, 1,
                                 ShmemBytes, streams_cuda[0], args, 0);
            // gpu_bsw::sequence_dna_reverse<<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
            //         strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
            //         gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            err = cuLaunchKernel(kernel2, sequences_per_stream + sequences_stream_leftover, 1, 1,
                                 newMin, 1, 1,
                                 ShmemBytes, streams_cuda[1], args_streamB, 0);
            // gpu_bsw::sequence_dna_reverse<<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
            //         strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
            //         gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
            //         gpu_data.scores_gpu + sequences_per_stream, matchScore, misMatchScore, startGap, extendGap);
            cudaErrchk(cudaGetLastError());

            asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);

                 alAbeg += stringsPerIt;
                 alBbeg += stringsPerIt;
                 alAend += stringsPerIt;
                 alBend += stringsPerIt;
                 top_scores_cpu += stringsPerIt;

            cudaStreamSynchronize (streams_cuda[0]);
            cudaStreamSynchronize (streams_cuda[1]);

        }  // for iterations end here

        cudaProfilerStop();

        auto end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends

    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
}// end of DNA kernel

void
gpu_bsw_driver::kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap, float factor)
{
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length

    short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                             13,7,8,9,0,11,10,12,2,0,14,5,
                             1,15,16,0,19,17,22,18,21};

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(deviceCount);// one OMP thread per GPU
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";

    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
      cudaGetDeviceProperties(&prop[i], 0);

    unsigned NBLOCKS             = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device     = NBLOCKS % deviceCount;
    unsigned max_per_device = alignmentsPerDevice + leftOver_device;
    
    initialize_alignments(alignments, totalAlignments); // pinned memory allocation
    auto start = NOW;
    size_t tot_mem_req_per_aln = maxReadSize + maxContigSize + 2 * sizeof(int) + 5 * sizeof(short);
    #pragma omp parallel
    {

      int my_cpu_id = omp_get_thread_num();
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }
      if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
        size_t gpu_mem_avail = get_tot_gpu_mem(myGPUid);
        unsigned max_alns_gpu = floor(((double)gpu_mem_avail*factor)/tot_mem_req_per_aln);
        unsigned max_alns_sugg = 20000;
        max_alns_gpu = max_alns_gpu > max_alns_sugg ? max_alns_sugg : max_alns_gpu;
        int       its    = (max_per_device>max_alns_gpu)?(ceil((double)max_per_device/max_alns_gpu)):1;
        std::cout<<"Mem (bytes) avail on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail<<"\n";
        std::cout<<"Mem (bytes) using on device "<<myGPUid<<":"<<(long unsigned)gpu_mem_avail*factor<<"\n";

      int BLOCKS_l = alignmentsPerDevice;
      if(my_cpu_id == deviceCount - 1)
          BLOCKS_l += leftOver_device;
      unsigned leftOvers    = BLOCKS_l % its;
      unsigned stringsPerIt = BLOCKS_l / its;
      gpu_alignments gpu_data(stringsPerIt + leftOvers); // gpu mallocs
      short *d_encoding_matrix, *d_scoring_matrix;
      cudaErrchk(cudaMalloc(&d_encoding_matrix, ENCOD_MAT_SIZE * sizeof(short)));
      cudaErrchk(cudaMalloc(&d_scoring_matrix, SCORE_MAT_SIZE * sizeof(short)));
      cudaErrchk(cudaMemcpy(d_encoding_matrix, encoding_matrix, ENCOD_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice));
      cudaErrchk(cudaMemcpy(d_scoring_matrix, scoring_matrix, SCORE_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice));

      short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
      short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
      short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
      short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice;  // memory on CPU for copying the results
      short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;

      unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
      cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
      unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
      cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));

      char *strA_d, *strB_d;
      cudaErrchk(cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char)));
      cudaErrchk(cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char)));

      char* strA;
      cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
      char* strB;
      cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));

      float total_packing = 0;

      auto start2 = NOW;
      std::cout<<"loop begin\n";
      for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
      {
          auto packing_start = NOW;
          int                                      blocksLaunched = 0;
          std::vector<std::string>::const_iterator beginAVec;
          std::vector<std::string>::const_iterator endAVec;
          std::vector<std::string>::const_iterator beginBVec;
          std::vector<std::string>::const_iterator endBVec;
          if(perGPUIts == its - 1)
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;  // so that each openmp thread has a copy of strings it needs to align
              blocksLaunched = stringsPerIt + leftOvers;
          }
          else
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) +  (perGPUIts + 1) * stringsPerIt;  // so that each openmp thread has a copy of strings it needs to align
              blocksLaunched = stringsPerIt;
          }

          std::vector<std::string> sequencesA(beginAVec, endAVec);
          std::vector<std::string> sequencesB(beginBVec, endBVec);
          unsigned running_sum = 0;
          int sequences_per_stream = (blocksLaunched) / NSTREAMS;
          int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
          unsigned half_length_A = 0;
          unsigned half_length_B = 0;

          auto start_cpu = NOW;

          for(int i = 0; i < sequencesA.size(); i++)
          {
              running_sum +=sequencesA[i].size();
              offsetA_h[i] = running_sum;//sequencesA[i].size();
              if(i == sequences_per_stream - 1){
                  half_length_A = running_sum;
                  running_sum = 0;
                }
          }
          unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];

          running_sum = 0;
          for(int i = 0; i < sequencesB.size(); i++)
          {
              running_sum +=sequencesB[i].size();
              offsetB_h[i] = running_sum; //sequencesB[i].size();
              if(i == sequences_per_stream - 1){
                half_length_B = running_sum;
                running_sum = 0;
              }
          }
          unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];

          auto end_cpu = NOW;
          std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

          total_time_cpu += cpu_dur.count();
          unsigned offsetSumA = 0;
          unsigned offsetSumB = 0;

          for(int i = 0; i < sequencesA.size(); i++)
          {
              char* seqptrA = strA + offsetSumA;
              memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
              char* seqptrB = strB + offsetSumB;
              memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
              offsetSumA += sequencesA[i].size();
              offsetSumB += sequencesB[i].size();
          }

          auto packing_end = NOW;
          std::chrono::duration<double> packing_dur = packing_end - packing_start;

          total_packing += packing_dur.count();

          asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, sequences_stream_leftover, streams_cuda);
          unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
          unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
          unsigned alignmentPad = 4 + (4 - totShmem % 4);
          size_t   ShmemBytes = totShmem + alignmentPad;
          if(ShmemBytes > 48000)
              cudaFuncSetAttribute(gpu_bsw::sequence_dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);

          gpu_bsw::sequence_aa_kernel<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
              strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
              gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu,
              openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          gpu_bsw::sequence_aa_kernel<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
              strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          // copyin back end index so that we can find new min
          asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);

          cudaStreamSynchronize (streams_cuda[0]);
          cudaStreamSynchronize (streams_cuda[1]);

          auto sec_cpu_start = NOW;
          int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
          auto sec_cpu_end = NOW;
          std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
          total_time_cpu += dur_sec_cpu.count();

          gpu_bsw::sequence_aa_reverse<<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
                  strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                  gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          gpu_bsw::sequence_aa_reverse<<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
                  strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
                  gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + sequences_per_stream,
                  gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaErrchk(cudaGetLastError());

          asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);

                alAbeg += stringsPerIt;
                alBbeg += stringsPerIt;
                alAend += stringsPerIt;
                alBend += stringsPerIt;
                top_scores_cpu += stringsPerIt;
		
		 cudaStreamSynchronize (streams_cuda[0]);
                 cudaStreamSynchronize (streams_cuda[1]);

      }  // for iterations end here

        auto end1  = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaErrchk(cudaFree(strA_d));
        cudaErrchk(cudaFree(strB_d));
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);

        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);

        std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends
    auto                          end  = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< diff.count() <<std::endl;
}// end of amino acids kernel
