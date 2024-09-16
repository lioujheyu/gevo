#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "options.hpp"
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

typedef unsigned long long int uint64_cu;

struct coord {
	int x,y,z;
};

inline int to_1d(int x, int y, int z,
					int sz_x, int sz_y, int sz_z) {
	return x + y * sz_x + z * sz_x * sz_y;
}

inline coord to_3d(int id, int sz_x, int sz_y, int sz_z) {
	int z = id / (sz_x * sz_y);
	int i = id % (sz_x * sz_y);
	int y = i / sz_x;
	int x = i % sz_x;
	coord c;
	c.x = x;
	c.y = y;
	c.z = z;
	return c;
}

struct GridPoint {

	//grid point data
	float virions;
	float nb_virions;
	float chemokine;
	float nb_chemokine;
	int num_neighbors;
	int epicell_status;
	int incubation_time_steps;
	int expressing_time_steps;
	int apoptotic_time_steps;

	//tcell data
	int id;
	int binding_period = -1;
	int tissue_time_steps = -1;
	
	//next step tcell data
	int from_id;
	int num_bound;
	int bound_from[27] = {-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1};

	int neighbor_count = 0;
	int neighborhood[27] = {-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1,-1,-1,-1,
							-1,-1};

};

class Tissue {
	public:
		//rng
		curandState* d_state;

		//sim data
		GridPoint* grid_points;

		//simulation world vars
		int num_points;
		int* num_circulating_tcells;
		int* tcells_generated;

		//useful stencil structures
		int num_adj;
		const static int neighborhood3D=27;
		int* loop_x;
		int* loop_y;
		int* loop_z;

		//parameters
		Options* params;

		//fields for holding simulation stats
		//these are device variables
		uint64_cu* total_virions;
		uint64_cu* total_tcells;
		uint64_cu* total_healthy;
		uint64_cu* total_incubating;
		uint64_cu* total_expressing;
		uint64_cu* total_apoptotic;
		uint64_cu* total_dead;
		float* total_chem;

		CUmodule module;
    	CUfunction device_age_tcells_kernel,
				   device_set_binding_tcells_kernel,
				   device_bind_tcells_kernel,
				   device_set_move_tcells_kernel,
				   device_move_tcells_kernel,
		           device_accumulate_kernel,
				   device_spread_chemokine_kernel,
				   device_reduce_values_kernel;
		FILE *f_total_virions;
		FILE *fout;

		//constructors and deconstructors
		Tissue(Options opt);
		~Tissue();

		//driver functions
		void dump_state(int iter);
		void reduce_values();
		void print_stats(int iter);
		void generate_tcells();
		void update_circulating_tcells();
		void update_tissue_tcells();
		void update_chemokine();
		void update_virions();
		void update_epicells();
		void initialize_infection(int x, int y, int z, float amount);
};