#include "tissue.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

inline void last_error(){
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess )
	{
		printf("#!!!CUDA Error: %s\n", cudaGetErrorString(err));  
	}
}

int blocks = 8;
int threads_per_block = 256;

// GPU Kernel declarations
__global__ void device_initialize_infection(int x,int y,int z,float amount,
	GridPoint* grid_points, Options* params);
__global__ void setup_grid_points(Options* params, GridPoint* grid_points, curandState* state,
								int* loop_x, int* loop_y, int* loop_z);
// __global__ void device_dump_state(Options* params, GridPoint* grid_points, int iter);
__global__ void setup_stencil(int* loop_x, int* loop_y, int* loop_z);
__global__ void device_generate_tcells(Options* params, GridPoint* grid_points, 
											int* num_circulating_tcells);
__global__ void device_update_circulating_tcells(Options* params, int* num_circulating_tcells,
													int* tcells_generated,
													GridPoint* grid_points, curandState* state, int num_xtravasing);
/**
* Begin T Cell Kernels
**/
__global__ void device_age_tcells(Options* params, GridPoint* grid_points);
__global__ void device_set_binding_tcells(Options* params, GridPoint* grid_points, curandState* state,
														int num_adj, int* loop_x, int* loop_y, int* loop_z);
__global__ void device_bind_tcells(Options* params, GridPoint* grid_points);
__global__ void device_set_move_tcells(Options* params, GridPoint* grid_points, curandState* state,
														int num_adj, int* loop_x, int* loop_y, int* loop_z);
__global__ void device_move_tcells(Options* params, GridPoint* grid_points);
/**
* End T Cell Kernels
**/

__global__ void device_spread_chemokine(Options* params,
										GridPoint* grid_points);
__global__ void device_update_virions(Options* params,
										GridPoint* grid_points);
__global__ void device_accumulate(Options* params, GridPoint* grid_points, int num_adj,
	int* loop_x, int* loop_y, int* loop_z);
__global__ void device_update_epicells(Options* params, GridPoint* grid_points, curandState* state);
__global__ void device_reduce_values(Options* params, GridPoint* grid_points,
									uint64_cu* total_virions,
									uint64_cu* total_tcells,
									uint64_cu* total_healthy,
									uint64_cu* total_incubating,
									uint64_cu* total_expressing,
									uint64_cu* total_apoptotic,
									uint64_cu* total_dead,
									float* total_chem);

// device functions
__device__ int device_to_1d(int x, int y, int z,
					int sz_x, int sz_y, int sz_z) {
	return x + y * sz_x + z * sz_x * sz_y;
}
__device__ coord device_to_3d(int id, int sz_x, int sz_y, int sz_z) {
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
__device__ int get_between(curandState* state, int min, int max){
	if(min == max){
		return min;
	}
	//Kirtus: NOTE: This function could be improved with more intelligent
	//rounding, will need to test a few different options.
	float roll = curand_uniform(state);
	int result = (int)(roll*(max - min)) + min;
	return result;
}
// A shuffled form of a is placed in the memory of b
// This is an implementation of the Fisher-Yates shuffle
// https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
__device__ void inplace_shuffle(curandState* state,
								int* a, int* b, int n) {
	//copy a to b
	for(int i = 0; i < n; i++) {
		b[i] = a[i];
	}
	//shuffle b in place
	for(int i = n - 1; i >= 1; i--) {
		int j = get_between(state, 0, i);
		int temp = b[j];
		b[j] = b[i];
		b[i] = temp;
	}
}

__device__ bool trial_success(curandState* state, float p){
	if(p > 1) return true;
	if(p == 0) return false;
	float roll = curand_uniform(state);
	if(roll < p){
		return true;
	}
	return false;
}

__device__ bool device_add_tcell(Options* params,
								GridPoint* grid_points,
								int* tcells_generated,
								int id, int life_time){
	if(grid_points[id].chemokine < params->min_chemokine) return false;
	int tcell_id = atomicAdd(tcells_generated,0);
	int prev_id = atomicCAS(&grid_points[id].id, -1, tcell_id);
	printf("### prev_id: %d\n", prev_id);
	if(prev_id != -1) return false;
	atomicAdd(tcells_generated, 1);
	grid_points[id].tissue_time_steps = life_time;

	return true;
}
__device__ bool check_bounds(int x, int y, int z, Options* params){
	if( x >= 0 && x < params->dim_x && 
		y >= 0 && y < params->dim_y && 
		z >= 0 && z < params->dim_z){ 
		return true;
	}
	return false;
}
__device__ bool check_bounds_2(int x, int y, int z, Options* params){ //c U1528, -p', 'U1529.OP1,U1525
	if( x >= 0 && x < params->dim_x && 
		/*y >= 0 &&*/ y < params->dim_y && //c U1528, -p', 'U1529.OP1,U1525
		z >= 0 && z < params->dim_z){ 
		return true;
	}
	return false;
}
__device__ bool device_try_bind_tcell(int source, int target, GridPoint* grid_points,
									int incubation_period, float max_binding_prob,
									curandState* state){
	int status = grid_points[target].epicell_status;
	if(status == 0 || status == 4) {
		return false;
	}
	double binding_prob = 0.0;
	if(status == 2 || status == 3 || grid_points[target].num_bound > 0) {
		binding_prob = max_binding_prob;
	} else {
		double scaling = 1.0 - (double)grid_points[target].incubation_time_steps/incubation_period;
		if(scaling < 0) scaling = 0;
		double prob = max_binding_prob*scaling;
		if(prob < max_binding_prob) {
			binding_prob = prob;
		} else {
			binding_prob = max_binding_prob;
		}
	}
	if(trial_success(state, binding_prob)) {

		int num_bound = atomicAdd(&grid_points[target].num_bound, 1);
		int result = atomicCAS(&grid_points[target].bound_from[num_bound], -1, source);

		if(result == -1){
			return true;
		}
		return false;
	}
	return false;
}

//useful host functions
coord host_to_3d(int id, int sz_x, int sz_y, int sz_z) {
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
bool h_trial_success(float p){
	float r = (float)rand() / ((float)RAND_MAX + 1);
	if(r<p){
		return true;
	}
	return false;
}

//GPU RNG kernels
__global__ void setup_curand(unsigned long long seed,
	unsigned long long offset, curandState* state);

// GPU Kernel declarations
// __global__ void setup_grid_points(Options* params,
// 	GridPoint* grid_points, curandState* state,
// 	int* loop_x, int* loop_y, int* loop_z) {
// 	int start = blockIdx.x*blockDim.x + threadIdx.x;
// 	int maximum = params->num_points;
// 	int step = blockDim.x*gridDim.x;
// 	int tid = threadIdx.x + blockDim.x * blockIdx.x;
// 	curandState local_state = state[tid];
// 	for(int i = start; i < maximum; i += step) {
// 		grid_points[i].virions = 0.0f;
// 		grid_points[i].nb_virions = 0.0f;
// 		grid_points[i].num_neighbors = 0;
// 		grid_points[i].epicell_status = 0;
// 		grid_points[i].incubation_time_steps = curand_poisson(&local_state, params->incubation_period);
// 		grid_points[i].expressing_time_steps = curand_poisson(&local_state, params->expressing_period);
// 		grid_points[i].apoptotic_time_steps = curand_poisson(&local_state, params->apoptosis_period);
// 		//define grid_point neighborhood
// 		grid_points[i].neighborhood[0] = i;
// 		grid_points[i].neighbor_count += 1;
// 		int idx = 1;
// 		for(int j = 0; j < 26; j++) {
// 			int zz = i / (params->dim_x * params->dim_y);
// 			int id = i % (params->dim_x * params->dim_y);
// 			int yy = id / params->dim_x;
// 			int xx = id % params->dim_x;
// 			xx += loop_x[j];
// 			yy += loop_y[j];
// 			zz += loop_z[j];
// 			if(check_bounds(xx,yy,zz,params)){
// 				int nb_id = device_to_1d(xx,yy,zz,
// 							params->dim_x, params->dim_y, params->dim_z);
// 				grid_points[i].neighborhood[idx] = nb_id;
// 				grid_points[i].neighbor_count += 1;
// 				idx++;
// 			}
// 		}
// 		//tcell data
// 		grid_points[i].binding_period = -1;
// 		grid_points[i].tissue_time_steps = -1;
// 		grid_points[i].id = -1;
// 		grid_points[i].from_id = -1;
// 		grid_points[i].num_bound = 0;
// 		for(int j = 0; j < 27; j++){
// 			grid_points[i].bound_from[j] = -1;
// 		}
// 	}
// 	state[tid] = local_state;
// }

//curand kernels
// __global__ void setup_curand(unsigned long long seed,
// 	unsigned long long offset, curandState* state){
// 	int id = threadIdx.x + blockIdx.x * blockDim.x;
// 	curand_init(seed, id, 0, &state[id]);
// }

// __global__ void device_generate_tcells(Options* params, GridPoint* grid_points, 
// 											int* num_circulating_tcells){
// 	atomicAdd(num_circulating_tcells, params->tcell_generation_rate);
// 	if(*num_circulating_tcells < 0) *num_circulating_tcells = 0;
// 	// printf("num_circulating_tcells = %d\n",*num_circulating_tcells);
// }

// __global__ void device_update_circulating_tcells(Options* params,
// 												int* num_circulating_tcells,
// 												int* tcells_generated,
// 												GridPoint* grid_points,
// 												curandState* state,
// 												int num_xtravasing) {

// 	int start = blockIdx.x*blockDim.x + threadIdx.x;
// 	int maximum = num_xtravasing;
// 	int step = blockDim.x*gridDim.x;
// 	int tid = threadIdx.x + blockDim.x * blockIdx.x;
// 	curandState local_state = state[tid];

// 	for(int i = start; i < maximum; i += step) {
// 		int x = get_between(&local_state, 0, params->dim_x-1);
// 		int y = get_between(&local_state, 0, params->dim_y-1);
// 		int z = get_between(&local_state, 0, params->dim_z-1);
// 		int id = device_to_1d(x,y,z,params->dim_x, params->dim_y, params->dim_z);
// 		int life_time = curand_poisson(&local_state, params->tcell_tissue_period);
// 		bool success = device_add_tcell(params,
// 							grid_points,
// 							tcells_generated,
// 							id, life_time);
//   		if(success) {
//   			printf("### Added tcell to %d,%d,%d\n",x,y,z);
//   			atomicSub(num_circulating_tcells, -1);
//   		} else {
//   			printf("### Couldn't add tcell to %d, %d,%d\n", x, y, z);
//   		}
// 	}
// 	state[tid] = local_state;
// }

// /**
// * Begin T Cell Kernels
// **/
__global__ void device_age_tcells(Options* params, GridPoint* grid_points){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = /*params->num_points*/threadIdx.x; //(('-c', 'U355'), ('-p', 'U358.OP1,U352'), ('-p', 'U381.OP1,U352'))
	int step = blockDim.x*gridDim.x;
	for(int i = start; i < maximum; i += step) {
		int tcell_id = grid_points[i].id;
		if(tcell_id == -1){
			continue;
		}

		// printf("###tcell_debug: id: %d, age: %d, binding_period: %d\n###at:%d\n\n",
		// 	grid_points[i].id,
		// 	grid_points[i].tissue_time_steps,
		// 	grid_points[i].binding_period,
		// 	i);

		grid_points[i].tissue_time_steps--;
		if(grid_points[i].tissue_time_steps <= 0) {
			//kill the t cell
			grid_points[i].id = -1;
			grid_points[i].binding_period = -1;
			grid_points[i].tissue_time_steps = -1;
			grid_points[i].from_id = -1;
		}
	}
}

__global__ void device_set_binding_tcells(Options* params, GridPoint* grid_points, curandState* state,
														int num_adj, int* loop_x, int* loop_y, int* loop_z){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState local_state = state[tid];
	for(int i = start; i < maximum; i += step) {

		int tcell_id = grid_points[i].id;
		if(tcell_id == -1){
			continue;
		}

		//decrement binding counter
		if(grid_points[i].binding_period != -1){
			grid_points[i].binding_period--;
			if(grid_points[i].binding_period <= 0){
				grid_points[i].binding_period = -1;
			}
		}else{
			int* neighbors = (int*)malloc(sizeof(int)*grid_points[i].neighbor_count);
			int* shuffledNeighbors = (int*)malloc(sizeof(int)*grid_points[i].neighbor_count);
			for(int j = 0; j < grid_points[i].neighbor_count; j++) {
				neighbors[j] = grid_points[i].neighborhood[j];
			}
			inplace_shuffle(&local_state,
				neighbors, shuffledNeighbors,grid_points[i].neighbor_count);
			for(int j = 0; j < grid_points[i].neighbor_count; j++) {
				int nb_id = shuffledNeighbors[j];
				device_try_bind_tcell(i, nb_id, grid_points,
										params->incubation_period,
										params->max_binding_prob, state);
			}
			free(neighbors);
			free(shuffledNeighbors);
		}
	}
	state[tid] = local_state;
}

__global__ void device_bind_tcells(Options* params, GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x; 
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	for(int i = start; i < maximum; i += step) { 
		if(grid_points[i].num_bound == 0){ //(('-c', 'U773'), ('-p', 'U774.OP0,U764'))
			continue;
		}

		for(int j = 0; j < grid_points[i].num_bound; j++){
			//induce apoptosis
			grid_points[i].epicell_status = 3;

			// set tcell binding period
			int bound_from_id = grid_points[i].bound_from[j];
			grid_points[bound_from_id].binding_period = params->tcell_binding_period;

			//reset bound from
			grid_points[i].bound_from[j] = -1;
		}
		grid_points[i].num_bound = 0;
	}
}

__global__ void device_set_move_tcells(Options* params, GridPoint* grid_points, curandState* state,
														int num_adj, int* loop_x, int* loop_y, int* loop_z){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState local_state = state[tid];
	for(int i = start; i < maximum; i += step) {
		int tcell_id = grid_points[i].id;
		if(tcell_id == -1){
			continue;
		}
		if(grid_points[i].binding_period != -1){
			continue;
		}

		//try to move
		int* neighbors = (int*)malloc(sizeof(int)*grid_points[i].neighbor_count);
		for(int j = 0; j < grid_points[i].neighbor_count; j++) {
			neighbors[j] = grid_points[i].neighborhood[j];
		}
		//try upto 5 times
		for(int k = 0; k < 5; k++){
			int try_idx = get_between(&local_state, 1,
				grid_points[i].neighbor_count);
			int nb_id = neighbors[try_idx];
			if(grid_points[nb_id].id != -1){
				continue;
			}

			int prev_id = atomicCAS(&grid_points[nb_id].from_id,
									-1, i);
			if(prev_id == -1){
				// printf("###move_debug! Should move from %d to %d\n",i,nb_id);
				break;
			}
		}
		free(neighbors);

	}
	state[tid] = local_state; // (('-c', 'U928'),)
}

__global__ void device_move_tcells(Options* params, GridPoint* grid_points){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	for(int i = start; i < maximum; i += step) {
		int from_id = grid_points[i].from_id;
		if(from_id != -1){
			// printf("###move_debug trying to move %d from %d to %d\n",
			// 	grid_points[from_id].id, from_id, i);
			//set new vals
			grid_points[i].id = grid_points[from_id].id;
			grid_points[i].binding_period = grid_points[from_id].binding_period;
			grid_points[i].tissue_time_steps = grid_points[from_id].tissue_time_steps;
			grid_points[i].from_id = -1;
			//clear old vals
			grid_points[from_id].id = -1;
			grid_points[from_id].binding_period = -1;
			grid_points[from_id].tissue_time_steps = -1;
			grid_points[from_id].from_id = -1;
		}
	}
}
// /**
// * End T Cell Kernels
// **/

__global__ void device_spread_chemokine(Options* params, GridPoint* grid_points){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	//spread
	for(int i = start; i < maximum; i += step) {
		float c = grid_points[i].chemokine;
		float nb_c = grid_points[i].nb_chemokine;
		int num_nb = grid_points[i].num_neighbors;

		float chemokine_diffused = c*params->chemokine_diffusion;
		float chemokine_left = c - chemokine_diffused;
		float avg_nb_chemokine = (chemokine_diffused + nb_c * params->chemokine_diffusion)/(num_nb + 1);

		grid_points[i].chemokine = chemokine_left + avg_nb_chemokine;
		grid_points[i].chemokine = (1.0 - params->chemokine_decay)*grid_points[i].chemokine;
		if(grid_points[i].chemokine < params->min_chemokine) grid_points[i].chemokine = 0.0;
		// grid_points[i].nb_chemokine = 0.0f; // -c U1342
	}
}

// __global__ void device_update_virions(Options* params, GridPoint* grid_points) {
// 	int start = blockIdx.x*blockDim.x + threadIdx.x;
// 	int maximum = params->num_points;
// 	int step = blockDim.x*gridDim.x;
// 	for(int i = start; i < maximum; i += step) {
// 		float v = grid_points[i].virions;
// 		float nb_v = grid_points[i].nb_virions;
// 		int num_nb = grid_points[i].num_neighbors;

// 		float virions_diffused = v*params->virion_diffusion;
// 		float virions_left = v - virions_diffused;
// 		float avg_nb_virions = (virions_diffused + nb_v * params->virion_diffusion)/(num_nb + 1);

// 		grid_points[i].virions = virions_left + avg_nb_virions;
// 		grid_points[i].virions = (1.0 - params->virion_clearance)*grid_points[i].virions;
// 		if(grid_points[i].virions < 0.0) grid_points[i].virions = 0.0;
// 		grid_points[i].nb_virions = 0.0f;
// 	}
// }

__global__ void device_accumulate(Options* params, GridPoint* grid_points, int num_adj,
	int* loop_x, int* loop_y, int* loop_z) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	for(int id = start; id < maximum; id += step) {
		grid_points[id].num_neighbors = 0;
		grid_points[id].nb_chemokine = 0.0f;
		// grid_points[id].nb_virions = 0.0f; // -c U1375
		for(int j = 0; j < num_adj; j++) {
			int zz = id / (params->dim_x * params->dim_y);
			int i = id % (params->dim_x * params->dim_y);
			int yy = i / params->dim_x;
			int xx = i % params->dim_x;

			xx += loop_x[j];
			yy += loop_y[j];
			zz += loop_z[j];

			int id_nb = device_to_1d(xx,yy,zz,
								params->dim_x, params->dim_y, params->dim_z);
			if(check_bounds(xx,yy,zz,params)) {
				int id_nb = device_to_1d(xx,yy,zz,
								params->dim_x, params->dim_y, params->dim_z);
				grid_points[id].nb_virions += grid_points[id_nb].virions;
				grid_points[id].num_neighbors += 1;
				// grid_points[id].nb_chemokine += grid_points[id_nb].chemokine;
			}
			if(check_bounds_2(xx,yy,zz,params)) { //c U1528, -p', 'U1529.OP1,U1525 and many many more
				grid_points[id].nb_chemokine += grid_points[id_nb].chemokine;
			}
		}
	}
	// for(int id = start; id < maximum; id += step) {
	// 	grid_points[id].nb_chemokine = 0.0f;
	// 	for(int j = 0; j < num_adj; j++) {
	// 		int zz = id / (params->dim_x * params->dim_y);
	// 		int i = id % (params->dim_x * params->dim_y);
	// 		int yy = i / params->dim_x;
	// 		int xx = i % params->dim_x;

	// 		xx += loop_x[j];
	// 		yy += loop_y[j];
	// 		zz += loop_z[j];

	// 		if(check_bounds_2(xx,yy,zz,params)) { //c U1528, -p', 'U1529.OP1,U1525
	// 			int id_nb = device_to_1d(xx,yy,zz,
	// 							params->dim_x, params->dim_y, params->dim_z);
	// 			grid_points[id].nb_chemokine += grid_points[id_nb].chemokine;
	// 		}
	// 	}
	// }
}

// __global__ void device_update_epicells(Options* params, GridPoint* grid_points, curandState* state){
// 	int start = blockIdx.x*blockDim.x + threadIdx.x;
// 	int maximum = params->num_points;
// 	int step = blockDim.x*gridDim.x;
// 	int tid = threadIdx.x + blockDim.x * blockIdx.x;
// 	curandState local_state = state[tid];
// 	for(int i = start; i < maximum; i += step) {
// 		bool produce_virions = false;
// 		switch(grid_points[i].epicell_status) {
// 			case 0:
// 				if(grid_points[i].virions > 0) {
// 					if(trial_success(&local_state, params->infectivity*grid_points[i].virions)) {
// 						grid_points[i].epicell_status = 1;
// 					}
// 				}
// 				break;
// 			case 1:
// 				grid_points[i].incubation_time_steps--;
// 				if(grid_points[i].incubation_time_steps <= 0) {
// 					grid_points[i].epicell_status = 2;
// 				}
// 				break;
// 			case 2:
// 				grid_points[i].expressing_time_steps--;
// 				if(grid_points[i].expressing_time_steps <= 0) {
// 					grid_points[i].epicell_status = 4;
// 				} else {
// 					produce_virions = true;
// 				}
// 				break;
// 			case 3:
// 				grid_points[i].apoptotic_time_steps--;
// 				if(grid_points[i].apoptotic_time_steps <= 0) {
// 					grid_points[i].epicell_status = 4;
// 				} else if (grid_points[i].incubation_time_steps==0) {
// 					produce_virions = true;
// 				}
// 				break;
// 			default: break;
// 		}
// 		if(produce_virions) {
// 			grid_points[i].virions += params->virion_production;
// 			if((grid_points[i].chemokine + params->chemokine_production) < 1.0) {
// 				grid_points[i].chemokine = (grid_points[i].chemokine + params->chemokine_production);
// 			} else {
// 				grid_points[i].chemokine = 1.0;
// 			}
// 		}
// 	}
// 	state[tid] = local_state;
// }

// __global__ void device_initialize_infection(int x,int y,int z,float amount,
// 											GridPoint* grid_points, Options* params){
// 	int id = device_to_1d(x,y,z,
// 								params->dim_x, params->dim_y, params->dim_z);
// 	grid_points[id].virions = amount;
// }

// __global__ void device_reduce_values(Options* params, GridPoint* grid_points,
// 									uint64_cu* total_virions,
// 									uint64_cu* total_tcells,
// 									uint64_cu* total_healthy,
// 									uint64_cu* total_incubating,
// 									uint64_cu* total_expressing,
// 									uint64_cu* total_apoptotic,
// 									uint64_cu* total_dead,
// 									float* total_chem) {
// 	int start = blockIdx.x*blockDim.x + threadIdx.x;
// 	int maximum = params->num_points;
// 	int step = blockDim.x*gridDim.x;
// 	// NOTE: These adds might need to be atomic!
// 	for(int i = start; i < maximum; i+=step) {
// 		GridPoint gp = grid_points[i];
// 		atomicAdd(total_virions, (uint64_cu)gp.virions);
// 		atomicAdd(total_chem, (float)gp.chemokine);
// 		if(gp.id != -1){
// 			atomicAdd(total_tcells, 1);
// 		}
// 		switch(gp.epicell_status) {
// 			case 0:
// 				atomicAdd(total_healthy, 1);
// 				break;
// 			case 1:
// 				atomicAdd(total_incubating, 1);
// 				break;
// 			case 2:
// 				atomicAdd(total_expressing, 1);
// 				break;
// 			case 3:
// 				atomicAdd(total_apoptotic, 1);
// 				break;
// 			case 4:
// 				atomicAdd(total_dead, 1);
// 			default: break;
// 		}
// 	}
// }

// __global__ void setup_stencil(int* loop_x, int* loop_y, int* loop_z){
// 	loop_x[0] = -1;
// 	loop_x[1] = -1;
// 	loop_x[2] = -1;
// 	loop_x[3] = -1;
// 	loop_x[4] = -1;
// 	loop_x[5] = -1;
// 	loop_x[6] = -1;
// 	loop_x[7] = -1;
// 	loop_x[8] = -1;
// 	loop_x[9] = 0;
// 	loop_x[10] = 0;
// 	loop_x[11] = 0;
// 	loop_x[12] = 0;
// 	loop_x[13] = 0;
// 	loop_x[14] = 0;
// 	loop_x[15] = 0;
// 	loop_x[16] = 0;
// 	loop_x[17] = 1;
// 	loop_x[18] = 1;
// 	loop_x[19] = 1;
// 	loop_x[20] = 1;
// 	loop_x[21] = 1;
// 	loop_x[22] = 1;
// 	loop_x[23] = 1;
// 	loop_x[24] = 1;
// 	loop_x[25] = 1;
// 	loop_y[0] = -1;
// 	loop_y[1] = -1;
// 	loop_y[2] = -1;
// 	loop_y[3] = 0;
// 	loop_y[4] = 0;
// 	loop_y[5] = 0;
// 	loop_y[6] = 1;
// 	loop_y[7] = 1;
// 	loop_y[8] = 1;
// 	loop_y[9] = -1;
// 	loop_y[10] = -1;
// 	loop_y[11] = -1;
// 	loop_y[12] = 0;
// 	loop_y[13] = 0;
// 	loop_y[14] = 1;
// 	loop_y[15] = 1;
// 	loop_y[16] = 1;
// 	loop_y[17] = -1;
// 	loop_y[18] = -1;
// 	loop_y[19] = -1;
// 	loop_y[20] = 0;
// 	loop_y[21] = 0;
// 	loop_y[22] = 0;
// 	loop_y[23] = 1;
// 	loop_y[24] = 1;
// 	loop_y[25] = 1;
// 	loop_z[0] = -1;
// 	loop_z[1] = 0;
// 	loop_z[2] = 1;
// 	loop_z[3] = -1;
// 	loop_z[4] = 0;
// 	loop_z[5] = 1;
// 	loop_z[6] = -1;
// 	loop_z[7] = 0;
// 	loop_z[8] = 1;
// 	loop_z[9] = -1;
// 	loop_z[10] = 0;
// 	loop_z[11] = 1;
// 	loop_z[12] = -1;
// 	loop_z[13] = 1;
// 	loop_z[14] = -1;
// 	loop_z[15] = 0;
// 	loop_z[16] = 1;
// 	loop_z[17] = -1;
// 	loop_z[18] = 0;
// 	loop_z[19] = 1;
// 	loop_z[20] = -1;
// 	loop_z[21] = 0;
// 	loop_z[22] = 1;
// 	loop_z[23] = -1;
// 	loop_z[24] = 0;
// 	loop_z[25] = 1;

// }