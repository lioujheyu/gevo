#include "driver.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include<bits/stdc++.h>
#include"utils.hpp"

using namespace std;

void proteinSampleRun(string refFile, string queFile, string out_file){
  long long int total_cells = 0;
  vector<string> G_sequencesA, G_sequencesB;
  string myInLine;
  int largestA = 0, largestB = 0, totSizeA = 0, totSizeB = 0;
  ifstream ref_file(refFile), quer_file(queFile);

  if(ref_file.is_open())
  {
      while(getline(ref_file, myInLine))
      {
        //  string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
          if(myInLine[0] == '>'){
            continue;
          }else{
            string seq = myInLine;
            G_sequencesA.push_back(seq);
            //std::cout<<"ref:"<<G_sequencesA.size()<<std::endl;
            totSizeA += seq.size();
            if(seq.size() > largestA)
            {
                largestA = seq.size();
            }

          }

      }
      ref_file.close();
  }


  if(quer_file.is_open())
  {
      while(getline(quer_file, myInLine))
      {
          //string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
          if(myInLine[0] == '>'){
            continue;
          }else{
            string seq = myInLine;
            G_sequencesB.push_back(seq);
          //  std::cout<<"que:"<<G_sequencesB.size()<<std::endl;
            totSizeB += seq.size();
            if(seq.size() > largestB)
            {
                largestB = seq.size();
            }

          }

      }
      quer_file.close();
  }




    short scores_matrix[] = {4 ,-1 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,0 ,-3 ,-2 ,0 ,-2 ,-1 ,0 ,-4 , -1 ,5 ,0 ,-2 ,-3 ,1 ,0 ,-2 ,0 ,-3 ,-2 ,2 ,-1 ,-3 ,-2 ,-1 ,-1 ,-3 ,-2 ,-3 ,-1 ,0 ,-1 ,-4 ,
    -2 ,0 ,6 ,1 ,-3 ,0 ,0 ,0 ,1 ,-3 ,-3 ,0 ,-2 ,-3 ,-2 ,1 ,0 ,-4 ,-2 ,-3 ,3 ,0 ,-1 ,-4 ,
    -2 ,-2 ,1 ,6 ,-3 ,0 ,2 ,-1 ,-1 ,-3 ,-4 ,-1 ,-3 ,-3 ,-1 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
    0 ,-3 ,-3 ,-3 ,9 ,-3 ,-4 ,-3 ,-3 ,-1 ,-1 ,-3 ,-1 ,-2 ,-3 ,-1 ,-1 ,-2 ,-2 ,-1 ,-3 ,-3 ,-2 ,-4 ,
    -1 ,1 ,0 ,0 ,-3 ,5 ,2 ,-2 ,0 ,-3 ,-2 ,1 ,0 ,-3 ,-1 ,0 ,-1 ,-2 ,-1 ,-2 ,0 ,3 ,-1 ,-4 ,
    -1 ,0 ,0 ,2 ,-4 ,2 ,5 ,-2 ,0 ,-3 ,-3 ,1 ,-2 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
    0 ,-2 ,0 ,-1 ,-3 ,-2 ,-2 ,6 ,-2 ,-4 ,-4 ,-2 ,-3 ,-3 ,-2 ,0 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-1 ,-4 ,
    -2 ,0 ,1 ,-1 ,-3 ,0 ,0 ,-2 ,8 ,-3 ,-3 ,-1 ,-2 ,-1 ,-2 ,-1 ,-2 ,-2 ,2 ,-3 ,0 ,0 ,-1 ,-4 ,
    -1 ,-3 ,-3 ,-3 ,-1 ,-3 ,-3 ,-4 ,-3 ,4 ,2 ,-3 ,1 ,0 ,-3 ,-2 ,-1 ,-3 ,-1 ,3 ,-3 ,-3 ,-1 ,-4 ,
    -1 ,-2 ,-3 ,-4 ,-1 ,-2 ,-3 ,-4 ,-3 ,2 ,4 ,-2 ,2 ,0 ,-3 ,-2 ,-1 ,-2 ,-1 ,1 ,-4 ,-3 ,-1 ,-4 ,
    -1 ,2 ,0 ,-1 ,-3 ,1 ,1 ,-2 ,-1 ,-3 ,-2 ,5 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,0 ,1 ,-1 ,-4 ,
    -1 ,-1 ,-2 ,-3 ,-1 ,0 ,-2 ,-3 ,-2 ,1 ,2 ,-1 ,5 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,1 ,-3 ,-1 ,-1 ,-4 ,
    -2 ,-3 ,-3 ,-3 ,-2 ,-3 ,-3 ,-3 ,-1 ,0 ,0 ,-3 ,0 ,6 ,-4 ,-2 ,-2 ,1 ,3 ,-1 ,-3 ,-3 ,-1 ,-4 ,
    -1 ,-2 ,-2 ,-1 ,-3 ,-1 ,-1 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-4 ,7 ,-1 ,-1 ,-4 ,-3 ,-2 ,-2 ,-1 ,-2 ,-4 ,
    1 ,-1 ,1 ,0 ,-1 ,0 ,0 ,0 ,-1 ,-2 ,-2 ,0 ,-1 ,-2 ,-1 ,4 ,1 ,-3 ,-2 ,-2 ,0 ,0 ,0 ,-4 ,
    0 ,-1 ,0 ,-1 ,-1 ,-1 ,-1 ,-2 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,5 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-4 ,
    -3 ,-3 ,-4 ,-4 ,-2 ,-2 ,-3 ,-2 ,-2 ,-3 ,-2 ,-3 ,-1 ,1 ,-4 ,-3 ,-2 ,11 ,2 ,-3 ,-4 ,-3 ,-2 ,-4 ,
    -2 ,-2 ,-2 ,-3 ,-2 ,-1 ,-2 ,-3 ,2 ,-1 ,-1 ,-2 ,-1 ,3 ,-3 ,-2 ,-2 ,2 ,7 ,-1 ,-3 ,-2 ,-1 ,-4 ,
    0 ,-3 ,-3 ,-3 ,-1 ,-2 ,-2 ,-3 ,-3 ,3 ,1 ,-2 ,1 ,-1 ,-2 ,-2 ,0 ,-3 ,-1 ,4 ,-3 ,-2 ,-1 ,-4 ,
    -2 ,-1 ,3 ,4 ,-3 ,0 ,1 ,-1 ,0 ,-3 ,-4 ,0 ,-3 ,-3 ,-2 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
    -1 ,0 ,0 ,1 ,-3 ,3 ,4 ,-2 ,0 ,-3 ,-3 ,1 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
    0 ,-1 ,-1 ,-1 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-2 ,0 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-4 ,
    -4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,1}; // blosum 62

    /*blposum 50 = {// 24 x 24 table
   //  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   B   Z   X   *
         5, -2, -1, -2, -1, -1, -1,  0, -2, -1, -2, -1, -1, -3, -1,  1,  0, -3, -2,  0, -2, -1, -1, -5,	// A
         -2,  7, -1, -2, -4,  1,  0, -3,  0, -4, -3,  3, -2, -3, -3, -1, -1, -3, -1, -3, -1,  0, -1, -5,	// R
         -1, -1,  7,  2, -2,  0,  0,  0,  1, -3, -4,  0, -2, -4, -2,  1,  0, -4, -2, -3,  5,  0, -1, -5,	// N
         -2, -2,  2,  8, -4,  0,  2, -1, -1, -4, -4, -1, -4, -5, -1,  0, -1, -5, -3, -4,  6,  1, -1, -5,	// D
         -1, -4, -2, -4, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -3, -3, -1, -5,	// C
         -1,  1,  0,  0, -3,  7,  2, -2,  1, -3, -2,  2,  0, -4, -1,  0, -1, -1, -1, -3,  0,  4, -1, -5,	// Q
         -1,  0,  0,  2, -3,  2,  6, -3,  0, -4, -3,  1, -2, -3, -1, -1, -1, -3, -2, -3,  1,  5, -1, -5,	// E
         0, -3,  0, -1, -3, -2, -3,  8, -2, -4, -4, -2, -3, -4, -2,  0, -2, -3, -3, -4, -1, -2, -1, -5,	// G
         -2,  0,  1, -1, -3,  1,  0, -2, 10, -4, -3,  0, -1, -1, -2, -1, -2, -3,  2, -4,  0,  0, -1, -5,	// H
         -1, -4, -3, -4, -2, -3, -4, -4, -4,  5,  2, -3,  2,  0, -3, -3, -1, -3, -1,  4, -4, -3, -1, -5,	// I
         -2, -3, -4, -4, -2, -2, -3, -4, -3,  2,  5, -3,  3,  1, -4, -3, -1, -2, -1,  1, -4, -3, -1, -5,	// L
         -1,  3,  0, -1, -3,  2,  1, -2,  0, -3, -3,  6, -2, -4, -1,  0, -1, -3, -2, -3,  0,  1, -1, -5,	// K
         -1, -2, -2, -4, -2,  0, -2, -3, -1,  2,  3, -2,  7,  0, -3, -2, -1, -1,  0,  1, -3, -1, -1, -5,	// M
         -3, -3, -4, -5, -2, -4, -3, -4, -1,  0,  1, -4,  0,  8, -4, -3, -2,  1,  4, -1, -4, -4, -1, -5,	// F
         -1, -3, -2, -1, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, -1, -1, -4, -3, -3, -2, -1, -1, -5,	// P
         1, -1,  1,  0, -1,  0, -1,  0, -1, -3, -3,  0, -2, -3, -1,  5,  2, -4, -2, -2,  0,  0, -1, -5,	// S
         0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  2,  5, -3, -2,  0,  0, -1, -1, -5, 	// T
         -3, -3, -4, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1,  1, -4, -4, -3, 15,  2, -3, -5, -2, -1, -5, 	// W
         -2, -1, -2, -3, -3, -1, -2, -3,  2, -1, -1, -2,  0,  4, -3, -2, -2,  2,  8, -1, -3, -2, -1, -5, 	// Y
         0, -3, -3, -4, -1, -3, -3, -4, -4,  4,  1, -3,  1, -1, -3, -2,  0, -3, -1,  5, -3, -3, -1, -5, 	// V
         -2, -1,  5,  6, -3,  0,  1, -1,  0, -4, -4,  0, -3, -4, -2,  0,  0, -5, -3, -3,  6,  1, -1, -5, 	// B
         -1,  0,  0,  1, -3,  4,  5, -2,  0, -3, -3,  1, -1, -4, -1,  0, -1, -2, -2, -3,  1,  5, -1, -5, 	// Z
         -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5, 	// X
         -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,  1 	// *
   };*/

    gpu_bsw_driver::alignment_results results_test;
    gpu_bsw_driver::kernel_driver_aa(G_sequencesB, G_sequencesA, &results_test, scores_matrix, -6, -1, 0.5);

  //  gpu_bsw_driver::verificationTest(resultFile, results_test.g_alAbeg, results_test.g_alBbeg, results_test.g_alAend, results_test.g_alBend);

  ofstream results_file(out_file);


  for(int k = 0; k < G_sequencesA.size(); k++){
        results_file<<results_test.top_scores[k]<<"\t"
                <<results_test.ref_begin[k]<<"\t"
                <<results_test.ref_end[k] - 1<<"\t"
                <<results_test.query_begin[k]<<"\t"
                <<results_test.query_end[k] - 1
                <<endl;
  }
  results_file.flush();
  results_file.close();

  for(int l = 0; l < G_sequencesA.size(); l++){
    total_cells += G_sequencesA.at(l).size()*G_sequencesB.at(l).size();
  }


    cout << "Total Cells:"<<total_cells<<endl;
}


void dnaSampleRun(string refFile, string queFile, string out_file){
  vector<string> G_sequencesA,
      G_sequencesB;

  string   myInLine;
  ifstream ref_file(refFile);
  ifstream quer_file(queFile);
  unsigned largestA = 0, largestB = 0;

  int totSizeA = 0, totSizeB = 0;

  if(ref_file.is_open())
  {
      while(getline(ref_file, myInLine))
      {
        //  string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
          if(myInLine[0] == '>'){
            continue;
          }else{
            string seq = myInLine;
            G_sequencesA.push_back(seq);
            //std::cout<<"ref:"<<G_sequencesA.size()<<std::endl;
            totSizeA += seq.size();
            if(seq.size() > largestA)
            {
                largestA = seq.size();
            }

          }

      }
      ref_file.close();
  }

  if(quer_file.is_open())
  {
      while(getline(quer_file, myInLine))
      {
          //string seq = myInLine.substr(myInLine.find(":") + 1, myInLine.size() - 1);
          if(myInLine[0] == '>'){
            continue;
          }else{
            string seq = myInLine;
            G_sequencesB.push_back(seq);
          //  std::cout<<"que:"<<G_sequencesB.size()<<std::endl;
            totSizeB += seq.size();
            if(seq.size() > largestB)
            {
                largestB = seq.size();
            }

          }

      }
      quer_file.close();
  }

  gpu_bsw_driver::alignment_results results_test;


  short scores[] = {1, -3, -3, -1};
  ofstream topscore_file("top_scores.out");
  ofstream results_file(out_file);

  gpu_bsw_driver::kernel_driver_dna(G_sequencesB, G_sequencesA,&results_test, scores, 0.5);
  for(int k = 0; k < G_sequencesA.size(); k++){
        topscore_file << results_test.top_scores[k] << endl;
        results_file <<results_test.ref_begin[k]     << "\t"
                     <<results_test.ref_end[k] - 1   << "\t"
                     <<results_test.query_begin[k]   << "\t"
                     <<results_test.query_end[k] - 1 << endl;
  }
  topscore_file.flush();
  topscore_file.close();
  results_file.flush();
  results_file.close();

  free_alignments(&results_test);
  long long int total_cells = 0;
  for(int l = 0; l < G_sequencesA.size(); l++){
    total_cells += G_sequencesA.at(l).size()*G_sequencesB.at(l).size();
  }
  cout <<"Total Cells:"<<total_cells<<endl;

  //gpu_bsw_driver::verificationTest(resultFile, results_test.g_alAbeg, results_test.g_alBbeg, results_test.g_alAend, results_test.g_alBend);

}

int
main(int argc, char* argv[])
{
  string in_arg = argv[1];

 if(in_arg == "aa"){
 	proteinSampleRun(argv[2], argv[3], argv[4]);
 }else{
 	dnaSampleRun(argv[2], argv[3], argv[4]);
 }

    return 0;
}
