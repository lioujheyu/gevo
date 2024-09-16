#include <kernel.hpp>

__inline__ __device__ short
gpu_bsw::warpReduceMax_with_index_reverse(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int   warpSize = 32;
    short myMax    = 0;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;
    myMax          = val;
    unsigned mask  = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);  // blockDim.x
    // unsigned newmask;
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {

        int tempVal = __shfl_down_sync(mask, val, offset);
        val     = max(val,tempVal);
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);

      //  if(threadIdx.x == 0)printf("index1:%d, index2:%d, max:%d\n", newInd, newInd2, val);
        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        else if((val == tempVal) ) // this is kind of redundant and has been done purely to match the results
                                    // with SSW to get the smallest alignment with highest score. Theoreticaly
                                    // all the alignmnts with same score are same.
        {
          if(newInd2 > ind2){
            ind = newInd;
            ind2 = newInd2;

          }
        }
    }
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}

__inline__ __device__ short
gpu_bsw::warpReduceMax_with_index(short val, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int   warpSize = 32;
    short myMax    = 0;
    short newInd   = 0;
    short newInd2  = 0;
    short ind      = myIndex;
    short ind2     = myIndex2;
    myMax          = val;
    unsigned mask  = __ballot_sync(0xffffffff, threadIdx.x < lengthSeqB);  // blockDim.x
    // unsigned newmask;
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {

        int tempVal = __shfl_down_sync(mask, val, offset);
        val     = max(val,tempVal);
        newInd  = __shfl_down_sync(mask, ind, offset);
        newInd2 = __shfl_down_sync(mask, ind2, offset);
        if(val != myMax)
        {
            ind   = newInd;
            ind2  = newInd2;
            myMax = val;
        }
        else if((val == tempVal) ) // this is kind of redundant and has been done purely to match the results
                                    // with SSW to get the smallest alignment with highest score. Theoreticaly
                                    // all the alignmnts with same score are same.
        {
          if(newInd < ind){
            ind = newInd;
            ind2 = newInd2;
          }
        }
    }
    myIndex  = ind;
    myIndex2 = ind2;
    val      = myMax;
    return val;
}



__device__ short
gpu_bsw::blockShuffleReduce_with_index_reverse(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax_with_index_reverse(myVal, myInd, myInd2, lengthSeqB);

    __syncthreads();
    if(laneId == 0)
        locTots[warpId] = myVal;
    if(laneId == 0)
        locInds[warpId] = myInd;
    if(laneId == 0)
        locInds2[warpId] = myInd2;
    __syncthreads();
    unsigned check =
        ((32 + blockDim.x - 1) / 32);  // mimicing the ceil function for floats
                                       // float check = ((float)blockDim.x / 32);
    if(threadIdx.x < check)  /////******//////
    {
        myVal  = locTots[threadIdx.x];
        myInd  = locInds[threadIdx.x];
        myInd2 = locInds2[threadIdx.x];
    }
    else
    {
        myVal  = 0;
        myInd  = -1;
        myInd2 = -1;
    }
    __syncthreads();

    if(warpId == 0)
    {
        myVal    = warpReduceMax_with_index_reverse(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}

__device__ short
gpu_bsw::blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, unsigned lengthSeqB)
{
    int              laneId = threadIdx.x % 32;
    int              warpId = threadIdx.x / 32;
    __shared__ short locTots[32];
    __shared__ short locInds[32];
    __shared__ short locInds2[32];
    short            myInd  = myIndex;
    short            myInd2 = myIndex2;
    myVal                   = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB);

    __syncthreads();
    if(laneId == 0)
        locTots[warpId] = myVal;
    if(laneId == 0)
        locInds[warpId] = myInd;
    if(laneId == 0)
        locInds2[warpId] = myInd2;
    __syncthreads();
    unsigned check =
        ((32 + blockDim.x - 1) / 32);  // mimicing the ceil function for floats
                                       // float check = ((float)blockDim.x / 32);
    if(threadIdx.x < check)  /////******//////
    {
        myVal  = locTots[threadIdx.x];
        myInd  = locInds[threadIdx.x];
        myInd2 = locInds2[threadIdx.x];
    }
    else
    {
        myVal  = 0;
        myInd  = -1;
        myInd2 = -1;
    }
    __syncthreads();

    if(warpId == 0)
    {
        myVal    = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB);
        myIndex  = myInd;
        myIndex2 = myInd2;
    }
    __syncthreads();
    return myVal;
}



__device__ __host__ short
           gpu_bsw::findMaxFour(short first, short second, short third, short fourth)
{
    short maxScore = 0;

    maxScore = max(first,second);
    maxScore = max(maxScore, third);
    maxScore = max(maxScore, fourth);

    return maxScore;
}

__global__ void
gpu_bsw::sequence_aa_kernel(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix)
{
  int block_Id  = blockIdx.x;
  int thread_Id = threadIdx.x;
  short laneId = threadIdx.x%32;
  short warpId = threadIdx.x/32;

  unsigned lengthSeqA;
  unsigned lengthSeqB;
  // local pointers
  char*    seqA;
  char*    seqB;
  char* longer_seq;

  extern __shared__ char is_valid_array[];
  char*                  is_valid = &is_valid_array[0];

// setting up block local sequences and their lengths.
  if(block_Id == 0)
  {
      lengthSeqA = prefix_lengthA[0];
      lengthSeqB = prefix_lengthB[0];
      seqA       = seqA_array;
      seqB       = seqB_array;
  }
  else
  {
      lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
      lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
      seqA       = seqA_array + prefix_lengthA[block_Id - 1];
      seqB       = seqB_array + prefix_lengthB[block_Id - 1];
  }
  // what is the max length and what is the min length
  unsigned maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
  unsigned minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

// shared memory space for storing longer of the two strings
  memset(is_valid, 0, minSize);
  is_valid += minSize;
  memset(is_valid, 1, minSize);
  is_valid += minSize;
  memset(is_valid, 0, minSize);

  char myColumnChar;
  // the shorter of the two strings is stored in thread registers
  if(lengthSeqA < lengthSeqB)
  {
    if(thread_Id < lengthSeqA)
      myColumnChar = seqA[thread_Id];  // read only once
    longer_seq = seqB;
  }
  else
  {
    if(thread_Id < lengthSeqB)
      myColumnChar = seqB[thread_Id];
    longer_seq = seqA;
  }

  __syncthreads(); // this is required here so that complete sequence has been copied to shared memory

  int   i            = 1;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

//initializing registers for storing diagonal values for three recent most diagonals (separate tables for
//H, E and F)
  short _curr_H = 0, _curr_F = 0, _curr_E = 0;
  short _prev_H = 0, _prev_F = 0, _prev_E = 0;
  short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
  short _temp_Val = 0;

 __shared__ short sh_prev_E[32]; // one such element is required per warp
 __shared__ short sh_prev_H[32];
 __shared__ short sh_prev_prev_H[32];

 __shared__ short local_spill_prev_E[1024];// each threads local spill,
 __shared__ short local_spill_prev_H[1024];
 __shared__ short local_spill_prev_prev_H[1024];

 __shared__ short sh_aa_encoding[ENCOD_MAT_SIZE];// length = 91
 __shared__ short sh_aa_scoring[SCORE_MAT_SIZE];

 int max_threads = blockDim.x;
 for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
    sh_aa_scoring[p] = scoring_matrix[p];
 }
 for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
   sh_aa_encoding[p] = encoding_matrix[p];
 }

  __syncthreads(); // to make sure all shmem allocations have been initialized

  for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
  {  // iterate for the number of anti-diagonals

      is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

       _temp_Val = _prev_H; // value exchange happens here to setup registers for next iteration
       _prev_H = _curr_H;
       _curr_H = _prev_prev_H;
       _prev_prev_H = _temp_Val;
       _curr_H = 0;

      _temp_Val = _prev_E;
      _prev_E = _curr_E;
      _curr_E = _prev_prev_E;
      _prev_prev_E = _temp_Val;
      _curr_E = 0;

      _temp_Val = _prev_F;
      _prev_F = _curr_F;
      _curr_F = _prev_prev_F;
      _prev_prev_F = _temp_Val;
      _curr_F = 0;


      if(laneId == 31)
      { // if you are the last thread in your warp then spill your values to shmem
        sh_prev_E[warpId] = _prev_E;
        sh_prev_H[warpId] = _prev_H;
        sh_prev_prev_H[warpId] = _prev_prev_H;
      }

      if(diag >= maxSize)
      { // if you are invalid in this iteration, spill your values to shmem
        local_spill_prev_E[thread_Id] = _prev_E;
        local_spill_prev_H[thread_Id] = _prev_H;
        local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
      }

      __syncthreads(); // this is needed so that all the shmem writes are completed.

      if(is_valid[thread_Id] && thread_Id < minSize)
      {
        unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));

        short fVal = _prev_F + extendGap;
        short hfVal = _prev_H + startGap;
        short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
        short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

        short eVal=0, heVal = 0;

        if(diag >= maxSize) // when the previous thread has phased out, get value from shmem
        {
          eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
          heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
        }
        else
        {
          eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
          heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
        }

         if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
         {
            eVal = 0;
            heVal = 0;
         }
        _curr_F = (fVal > hfVal) ? fVal : hfVal;
        _curr_E = (eVal > heVal) ? eVal : heVal;

        short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
        short final_prev_prev_H = 0;
        if(diag >= maxSize)
        {
          final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
        }
        else
        {
          final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
        }


        if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

        short mat_index_q = sh_aa_encoding[(int)longer_seq[i-1]];//encoding_matrix
        short mat_index_r = sh_aa_encoding[(int)myColumnChar];

        short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

        short diag_score = final_prev_prev_H + add_score;

        _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);
        thread_max_i = (thread_max >= _curr_H) ? thread_max_i : i;
        thread_max_j = (thread_max >= _curr_H) ? thread_max_j : thread_Id + 1;
        thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
        i++;
     }

    __syncthreads(); // why do I need this? commenting it out breaks it

  }
  __syncthreads();

  thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j,
                                  minSize);  // thread 0 will have the correct values




  if(thread_Id == 0)
  {


      if(lengthSeqA < lengthSeqB)
      {
        seqB_align_end[block_Id] = thread_max_i;
        seqA_align_end[block_Id] = thread_max_j;
        top_scores[block_Id] = thread_max;
    }
      else
      {
      seqA_align_end[block_Id] = thread_max_i;
      seqB_align_end[block_Id] = thread_max_j;
      top_scores[block_Id] = thread_max;
      }

  }
}

__global__ void
gpu_bsw::sequence_aa_reverse(char* seqA_array, char* seqB_array, unsigned* prefix_lengthA,
                    unsigned* prefix_lengthB, short* seqA_align_begin, short* seqA_align_end,
                    short* seqB_align_begin, short* seqB_align_end, short* top_scores, short startGap, short extendGap, short* scoring_matrix, short* encoding_matrix){

      int block_Id  = blockIdx.x;
      int thread_Id = threadIdx.x;
      short laneId = threadIdx.x%32;
      short warpId = threadIdx.x/32;
      // local pointers
      char*    seqA;
      char*    seqB;

      extern __shared__ char is_valid_array[];
      char*                  is_valid = &is_valid_array[0];
      char* longer_seq;

  // setting up block local sequences and their lengths.
      if(block_Id == 0)
      {
          seqA       = seqA_array;
          seqB       = seqB_array;
      }
      else
      {
          seqA       = seqA_array + prefix_lengthA[block_Id - 1];
          seqB       = seqB_array + prefix_lengthB[block_Id - 1];
      }
      int newlengthSeqA = seqA_align_end[block_Id];
      int newlengthSeqB = seqB_align_end[block_Id];
      unsigned maxSize = newlengthSeqA > newlengthSeqB ? newlengthSeqA : newlengthSeqB;
      unsigned minSize = newlengthSeqA < newlengthSeqB ? newlengthSeqA : newlengthSeqB;
      char myColumnChar;
      is_valid = &is_valid_array[0]; //reset is_valid array for second iter
      memset(is_valid, 0, minSize);
      is_valid += minSize;
      memset(is_valid, 1, minSize);
      is_valid += minSize;
      memset(is_valid, 0, minSize);
      __syncthreads(); // this is required because shmem writes

  //check if the new length of A is larger than B, if so then place the B string in registers and A in myLocString, make sure we dont do redundant copy by checking which string is located in myLocString before

      if(newlengthSeqA < newlengthSeqB)
      {
        if(thread_Id < newlengthSeqA)
        {
          myColumnChar = seqA[(newlengthSeqA - 1) - thread_Id];  // read only once
          longer_seq = seqB;
          }
      }
      else
      {
        if(thread_Id < newlengthSeqB)
        {
         myColumnChar = seqB[(newlengthSeqB - 1) - thread_Id];
           longer_seq = seqA;
        }
      }


      __syncthreads(); // this is required  because sequence has been re-written in shmem


      int   i            = 1;
      short thread_max   = 0; // to maintain the thread max score
      short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
      short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

    //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
    //H, E and F)
      short _curr_H = 0, _curr_F = 0, _curr_E = 0;
      short _prev_H = 0, _prev_F = 0, _prev_E = 0;
      short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
      short _temp_Val = 0;

     __shared__ short sh_prev_E[32]; // one such element is required per warp
     __shared__ short sh_prev_H[32];
     __shared__ short sh_prev_prev_H[32];

     __shared__ short local_spill_prev_E[1024];// each threads local spill,
     __shared__ short local_spill_prev_H[1024];
     __shared__ short local_spill_prev_prev_H[1024];

     __shared__ short sh_aa_encoding[ENCOD_MAT_SIZE];// length = 91
     __shared__ short sh_aa_scoring[SCORE_MAT_SIZE];

     int max_threads = blockDim.x;

     for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
       sh_aa_scoring[p] = scoring_matrix[p];
     }
     for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
       sh_aa_encoding[p] = encoding_matrix[p];
     }


      __syncthreads(); // this is required because shmem has been updated
      for(int diag = 0; diag <  newlengthSeqA + newlengthSeqB - 1; diag++)
      {
        is_valid = is_valid - (diag < minSize || diag >= maxSize);

        _temp_Val = _prev_H;
      	_prev_H = _curr_H;
      	_curr_H = _prev_prev_H;
      	_prev_prev_H = _temp_Val;

      	_curr_H = 0;


        _temp_Val = _prev_E;
        _prev_E = _curr_E;
        _curr_E = _prev_prev_E;
        _prev_prev_E = _temp_Val;
        _curr_E = 0;

        _temp_Val = _prev_F;
        _prev_F = _curr_F;
        _curr_F = _prev_prev_F;
        _prev_prev_F = _temp_Val;
      	_curr_F = 0;

        if(laneId == 31)
        {
          sh_prev_E[warpId] = _prev_E;
          sh_prev_H[warpId] = _prev_H;
    		  sh_prev_prev_H[warpId] = _prev_prev_H;
        }

        if(diag>= maxSize)
        { // if you are invalid in this iteration, spill your values to shmem
          local_spill_prev_E[thread_Id] = _prev_E;
          local_spill_prev_H[thread_Id] = _prev_H;
          local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
        }

        __syncthreads();

        if(is_valid[thread_Id] && thread_Id < minSize)
        {

      	    unsigned mask  = __ballot_sync(__activemask(), (is_valid[thread_Id] &&( thread_Id < minSize)));


      	    short fVal = _prev_F + extendGap;
      	    short hfVal = _prev_H + startGap;
      	    short valeShfl = __shfl_sync(mask, _prev_E, laneId- 1, 32);
      	    short valheShfl = __shfl_sync(mask, _prev_H, laneId - 1, 32);

            short eVal=0;
            short heVal = 0;

            if(diag >= maxSize)
            {
                eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
            }
            else
            {
              eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
            }

            if(diag >= maxSize)
            {
                heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
            }
            else
            {
              heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
            }

            if(warpId == 0 && laneId == 0)
            {
              eVal = 0;
              heVal = 0;
            }
            _curr_F = (fVal > hfVal) ? fVal : hfVal;
        		_curr_E = (eVal > heVal) ? eVal : heVal;
            short testShufll = __shfl_sync(mask, _prev_prev_H, laneId - 1, 32);
      	    short final_prev_prev_H =0;

            if(diag >= maxSize)
             {
               final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
             }
            else
             {
               final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
             }

            if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

            short mat_index_q = sh_aa_encoding[(int)longer_seq[maxSize-i]];
            short mat_index_r = sh_aa_encoding[(int)myColumnChar];
            short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

            short diag_score = final_prev_prev_H + add_score;
            _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);

            thread_max_i = (thread_max >= _curr_H) ? thread_max_i : maxSize - i ;//i;// begin_A (longer string)
            thread_max_j = (thread_max >= _curr_H) ? thread_max_j : minSize - thread_Id -1; // begin_B (shorter string)
            thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
            i++;

        }
          __syncthreads();

        }
        __syncthreads();




        thread_max = blockShuffleReduce_with_index_reverse(thread_max, thread_max_i, thread_max_j,
                                        minSize);  // thread 0 will have the correct values
        if(thread_Id == 0)
        {
          if(newlengthSeqA < newlengthSeqB)
          {
            seqB_align_begin[block_Id] = /*newlengthSeqB - */(thread_max_i);
            seqA_align_begin[block_Id] = /*newlengthSeqA */ (thread_max_j);
          }
          else
          {
            seqA_align_begin[block_Id] = /*newlengthSeqA - */(thread_max_i);
            seqB_align_begin[block_Id] = /*newlengthSeqB -*/ (thread_max_j);
          }
        }
        __syncthreads();

}