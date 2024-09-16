#include "tissue.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <assert.h>
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
	if(prev_id != -1) return false;
	atomicAdd(tcells_generated, 1);
	grid_points[id].tissue_time_steps = life_time;

	return true;
}
__device__ bool check_bounds(int x, int y, int z, Options* params){
	if(x >= 0 && x < params->dim_x &&
		y >= 0 && y < params->dim_y &&
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

Tissue::Tissue(Options opt) {
	Options* h_params = &opt;
	num_points = opt.dim_x*opt.dim_y*opt.dim_z;

	//allocate grid
	cudaMalloc((void**)&grid_points, num_points*sizeof(GridPoint));
	last_error();

	//copy params to device
	cudaMalloc((void**)&params, sizeof(Options));
	last_error();
	cudaMemcpy(params, h_params, sizeof(Options), cudaMemcpyHostToDevice);
	last_error();

	//set up RNG
	cudaMalloc((void**)&d_state, threads_per_block*blocks*sizeof(curandState));
	last_error();
	setup_curand<<<blocks, threads_per_block>>>((unsigned long long)opt.seed,
										0, d_state);
	last_error();

	//set up stencil structure on device
	num_adj = 26;
	cudaMalloc((void**)&loop_x, sizeof(int)*num_adj);
	last_error();
	cudaMalloc((void**)&loop_y, sizeof(int)*num_adj);
	last_error();
	cudaMalloc((void**)&loop_z, sizeof(int)*num_adj);
	last_error();
	setup_stencil<<<1,1>>>(loop_x, loop_y, loop_z);
	last_error();

	//set up grid points
	setup_grid_points<<<blocks, threads_per_block>>>(params, grid_points, d_state,
		loop_x, loop_y, loop_z);
	last_error();

	//set up tcell fields
	int z = 0;
	cudaMalloc((void**)&num_circulating_tcells, sizeof(int));
	last_error();
	cudaMemcpy(num_circulating_tcells,
				&z,
				sizeof(int),
				cudaMemcpyHostToDevice);
	last_error();
	cudaMalloc((void**)&tcells_generated, sizeof(int));
	last_error();
	cudaMemcpy(tcells_generated,
				&z,
				sizeof(int),
				cudaMemcpyHostToDevice);
	last_error();

	//set up reduction fields
	uint64_cu zz = 0;
	cudaMalloc((void**)&total_virions, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_virions,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMalloc((void**)&total_healthy, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_healthy,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMalloc((void**)&total_incubating, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_incubating,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMalloc((void**)&total_expressing, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_expressing,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMalloc((void**)&total_apoptotic, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_apoptotic,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMalloc((void**)&total_tcells, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_tcells,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMalloc((void**)&total_dead, sizeof(uint64_cu)); last_error();
	cudaMemcpy(total_dead,
				&zz,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	float fz = 0.0;
	cudaMalloc((void**)&total_chem, sizeof(float)); last_error();
	cudaMemcpy(total_chem,
				&fz,
				sizeof(float),
				cudaMemcpyHostToDevice); last_error();
	//set up host rng
	srand(opt.seed);

	unsigned int err;
	err = cuModuleLoad(&module, "gevo.ptx");
	assert(err==0 && "cuModuleLoad error");

	err = cuModuleGetFunction(&device_age_tcells_kernel, module,
							  "_Z17device_age_tcellsP7OptionsP9GridPoint");
	assert(err==0 && "cuModuleGetFunction device_age_tcells_kernel error");
	err = cuModuleGetFunction(&device_set_binding_tcells_kernel, module,
							  "_Z25device_set_binding_tcellsP7OptionsP9GridPointP17curandStateXORWOWiPiS5_S5_");
	assert(err==0 && "cuModuleGetFunction device_set_binding_tcells error");
	err = cuModuleGetFunction(&device_bind_tcells_kernel, module,
							  "_Z18device_bind_tcellsP7OptionsP9GridPoint");
	assert(err==0 && "cuModuleGetFunction device_bind_tcells error");
	err = cuModuleGetFunction(&device_set_move_tcells_kernel, module,
							  "_Z22device_set_move_tcellsP7OptionsP9GridPointP17curandStateXORWOWiPiS5_S5_");
	assert(err==0 && "cuModuleGetFunction device_set_move_tcells error");
	err = cuModuleGetFunction(&device_move_tcells_kernel, module,
							  "_Z18device_move_tcellsP7OptionsP9GridPoint");
	assert(err==0 && "cuModuleGetFunction device_move_tcells error");
	err = cuModuleGetFunction(&device_accumulate_kernel, module,
							  "_Z17device_accumulateP7OptionsP9GridPointiPiS3_S3_");
	assert(err==0 && "cuModuleGetFunction device_accumulate error");
	err = cuModuleGetFunction(&device_spread_chemokine_kernel, module,
							  "_Z23device_spread_chemokineP7OptionsP9GridPoint");
	assert(err==0 && "cuModuleGetFunction device_spread_chemokine error");
	err = cuModuleGetFunction(&device_reduce_values_kernel, module,
							  "_Z20device_reduce_valuesP7OptionsP9GridPointPyS3_S3_S3_S3_S3_S3_Pf");
	assert(err==0 && "cuModuleGetFunction device_reduce_values error");
	fout = fopen("output", "w");
}

Tissue::~Tissue(){
	fclose(fout);
	cudaFree(grid_points);
	cudaFree(params);
	cudaFree(loop_x);
	cudaFree(loop_y);
	cudaFree(loop_z);
	cudaFree(num_circulating_tcells);
	cudaFree(tcells_generated);
	cudaFree(total_virions);
	cudaFree(total_tcells);
	cudaFree(total_healthy);
	cudaFree(total_incubating);
	cudaFree(total_expressing);
	cudaFree(total_apoptotic);
	cudaFree(total_dead);
	cudaFree(total_chem);
}

//Host functions
void Tissue::initialize_infection(int x, int y, int z, float amount){
	device_initialize_infection<<<1,1>>>(x,y,z,amount,grid_points,params);
}

void Tissue::dump_state(int iter){
	// copy the data from the device to host
	Options* h_params = new Options;
	cudaMemcpy(h_params,
				params,
				sizeof(Options),
				cudaMemcpyDeviceToHost);
	last_error();

	GridPoint* h_grid_points = new GridPoint[num_points];
	cudaMemcpy(h_grid_points,
				grid_points,
				sizeof(GridPoint)*num_points,
				cudaMemcpyDeviceToHost);
	last_error();

	for(int i = 0; i < num_points; i += h_params->sample_resolution){
		GridPoint gp = h_grid_points[i];
		coord c = host_to_3d(i, h_params->dim_x, h_params->dim_y, h_params->dim_z);
		printf("%d,%d,%d,%d,%d,%f,%f,%d,%d\n",iter, i, c.x, c.y, c.z,
			gp.virions,
			gp.chemokine,
			gp.tissue_time_steps,
			gp.epicell_status);
	}

	//clear host memory
	delete h_params;
	delete[] h_grid_points;
}

void Tissue::generate_tcells(){
	device_generate_tcells<<<1,1>>>(params, grid_points, num_circulating_tcells);
}

void Tissue::update_circulating_tcells(){
	// copy relevant data from the device to host
	int* h_num_circulating_tcells = new int;
	Options* h_params = new Options;

	cudaMemcpy(h_num_circulating_tcells,
				num_circulating_tcells,
				sizeof(int),
				cudaMemcpyDeviceToHost);
	last_error();
	cudaMemcpy(h_params, params, sizeof(Options), cudaMemcpyDeviceToHost);
	last_error();

	double portion_dying = (double)(*h_num_circulating_tcells)/h_params->tcell_vascular_period;
	int num_dying = floor(portion_dying);
	if(h_trial_success((float)(portion_dying - num_dying))){
		num_dying++;
	}
	*h_num_circulating_tcells -= num_dying;
	if (*h_num_circulating_tcells < 0) *h_num_circulating_tcells = 0;
	long long whole_lung_volume = (long long)h_params->whole_lung_x *
                              (long long)h_params->whole_lung_y *
                              (long long)h_params->whole_lung_z;
	double extravasate_fraction = ((double)h_params->num_points)/(double)whole_lung_volume;
	double portion_xtravasing = extravasate_fraction * (*h_num_circulating_tcells);
  	int num_xtravasing = floor(portion_xtravasing);
  	if(h_trial_success((float)(portion_xtravasing-num_xtravasing))){
  		num_xtravasing++;
  	}

  	//copy relevant data back to device
  	cudaMemcpy(num_circulating_tcells,
				h_num_circulating_tcells,
				sizeof(int),
				cudaMemcpyHostToDevice);
	last_error();

	// printf("update_circulating_tcells: num_circulating: %d, num_xtravasing: %d\n",
		// *h_num_circulating_tcells, num_xtravasing);

	// printf("%d\n", num_xtravasing);

	device_update_circulating_tcells<<<blocks,threads_per_block>>>(params,
		num_circulating_tcells,
		tcells_generated,
		grid_points,
		d_state, num_xtravasing);

	//clear copied memory on host
	delete h_num_circulating_tcells;
	delete h_params;

}

void Tissue::update_tissue_tcells(){

	unsigned int err = 0;
	void* args0[] = {&params,
				&grid_points
				};

	// device_age_tcells<<<blocks,threads_per_block>>>(params, grid_points);
	err = cuLaunchKernel(device_age_tcells_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args0, 0);
	assert(err==0 && "device_age_tcells_kernel error");
	// device_set_binding_tcells<<<blocks,threads_per_block>>>(params, grid_points, d_state,
	// 														num_adj, loop_x, loop_y, loop_z);
	void* args1[] = {&params,
				&grid_points,
				&d_state,
				&num_adj,
				&loop_x,
				&loop_y,
				&loop_z
				};
	err = cuLaunchKernel(device_set_binding_tcells_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args1, 0);
	assert(err==0 && "device_set_binding_tcells_kernel error");
	// device_bind_tcells<<<blocks,threads_per_block>>>(params, grid_points);
	err = cuLaunchKernel(device_bind_tcells_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args0, 0);
	assert(err==0 && "device_bind_tcells_kernel error");
	// device_set_move_tcells<<<blocks,threads_per_block>>>(params, grid_points, d_state,
	// 														num_adj, loop_x, loop_y, loop_z);
	err = cuLaunchKernel(device_set_move_tcells_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args1, 0);
	assert(err==0 && "device_set_move_tcells_kernel error");
	// device_move_tcells<<<blocks,threads_per_block>>>(params, grid_points);
	err = cuLaunchKernel(device_move_tcells_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args0, 0);
	assert(err==0 && "device_move_tcells_kernel error");
}

void Tissue::update_chemokine(){
	unsigned int err = 0;
	void* args0[] = {&params,
				&grid_points,
				};
	void* args1[] = {&params,
				&grid_points,
				&num_adj,
				&loop_x,
				&loop_y,
				&loop_z
				};

	// device_accumulate<<<blocks,threads_per_block>>>(params, grid_points, num_adj,
	// 	loop_x, loop_y, loop_z);
	err = cuLaunchKernel(device_accumulate_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args1, 0);
	assert(err==0 && "device_accumulate_kernel error");
	// // device_update_chemokine<<<blocks,threads_per_block>>>(params, grid_points);
	// device_spread_chemokine<<<blocks,threads_per_block>>>(params, grid_points);
	err = cuLaunchKernel(device_spread_chemokine_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args0, 0);
	assert(err==0 && "device_spread_chemokine_kernel error");
}

void Tissue::update_virions(){
	device_update_virions<<<blocks,threads_per_block>>>(params, grid_points);
}

void Tissue::update_epicells(){
	device_update_epicells<<<blocks,threads_per_block>>>(params, grid_points, d_state);
}

void Tissue::reduce_values(){
	//reset vals to 0
	uint64_cu z = 0;
	cudaMemcpy(total_virions,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMemcpy(total_healthy,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMemcpy(total_incubating,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMemcpy(total_expressing,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMemcpy(total_apoptotic,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMemcpy(total_tcells,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	cudaMemcpy(total_dead,
				&z,
				sizeof(uint64_cu),
				cudaMemcpyHostToDevice); last_error();
	float zz = 0.0;
	cudaMemcpy(total_chem,
				&zz,
				sizeof(float),
				cudaMemcpyHostToDevice); last_error();
				
	unsigned int err = 0;
	void* args0[] = {&params,
				&grid_points,
				&total_virions,
				&total_tcells,
				&total_healthy,
				&total_incubating,
				&total_expressing,
				&total_apoptotic,
				&total_dead,
				&total_chem
				};
	// device_reduce_values<<<blocks,threads_per_block>>>(params,
	// 								grid_points,
	// 								total_virions,
	// 								total_tcells,
	// 								total_healthy,
	// 								total_incubating,
	// 								total_expressing,
	// 								total_apoptotic,
	// 								total_dead,
	// 								total_chem);
	err = cuLaunchKernel(device_reduce_values_kernel,
					   blocks, 1, 1,
				       threads_per_block, 1, 1,
				   	   0, 0, args0, 0);
	assert(err==0 && "device_reduce_values_kernel error");
}

void Tissue::print_stats(int iter){
	uint64_cu* h_total_virions = new uint64_cu;
	uint64_cu* h_total_tcells = new uint64_cu;
	uint64_cu* h_total_healthy = new uint64_cu;
	uint64_cu* h_total_incubating = new uint64_cu;
	uint64_cu* h_total_expressing = new uint64_cu;
	uint64_cu* h_total_apoptotic = new uint64_cu;
	uint64_cu* h_total_dead = new uint64_cu;
	float* h_total_chem = new float;
	int* h_num_circulating_tcells = new int;
	cudaMemcpy(h_total_virions,
				total_virions,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_healthy,
				total_healthy,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_incubating,
				total_incubating,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_expressing,
				total_expressing,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_apoptotic,
				total_apoptotic,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_tcells,
				total_tcells,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_dead,
				total_dead,
				sizeof(uint64_cu),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_num_circulating_tcells,
				num_circulating_tcells,
				sizeof(int),
				cudaMemcpyDeviceToHost); last_error();
	cudaMemcpy(h_total_chem,
				total_chem,
				sizeof(float),
				cudaMemcpyDeviceToHost); last_error();
	printf("%d,%llu,%d,%llu,%llu,%llu,%llu,%llu,%llu,%f\n",iter,
					*h_total_virions,
					*h_num_circulating_tcells,
					*h_total_tcells,
					*h_total_healthy,
					*h_total_incubating,
					*h_total_expressing,
					*h_total_apoptotic,
					*h_total_dead,
					*h_total_chem);
	fprintf(fout,"%d %llu %d %llu %llu %llu %llu %llu %llu %f\n",iter,
					*h_total_virions,
					*h_num_circulating_tcells,
					*h_total_tcells,
					*h_total_healthy,
					*h_total_incubating,
					*h_total_expressing,
					*h_total_apoptotic,
					*h_total_dead,
					*h_total_chem);
	delete h_total_virions;
	delete h_total_healthy;
	delete h_total_incubating;
	delete h_total_expressing;
	delete h_total_apoptotic;
	delete h_total_tcells;
	delete h_total_dead;
	delete h_num_circulating_tcells;
	delete h_total_chem;
}

// GPU Kernel declarations
__global__ void setup_grid_points(Options* params,
	GridPoint* grid_points, curandState* state,
	int* loop_x, int* loop_y, int* loop_z) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState local_state = state[tid];
	for(int i = start; i < maximum; i += step) {
		grid_points[i].virions = 0.0f;
		grid_points[i].nb_virions = 0.0f;
		grid_points[i].num_neighbors = 0;
		grid_points[i].epicell_status = 0;
		grid_points[i].incubation_time_steps = curand_poisson(&local_state, params->incubation_period);
		grid_points[i].expressing_time_steps = curand_poisson(&local_state, params->expressing_period);
		grid_points[i].apoptotic_time_steps = curand_poisson(&local_state, params->apoptosis_period);
		//define grid_point neighborhood
		grid_points[i].neighborhood[0] = i;
		grid_points[i].neighbor_count += 1;
		int idx = 1;
		for(int j = 0; j < 26; j++) {
			int zz = i / (params->dim_x * params->dim_y);
			int id = i % (params->dim_x * params->dim_y);
			int yy = id / params->dim_x;
			int xx = id % params->dim_x;
			xx += loop_x[j];
			yy += loop_y[j];
			zz += loop_z[j];
			if(check_bounds(xx,yy,zz,params)){
				int nb_id = device_to_1d(xx,yy,zz,
							params->dim_x, params->dim_y, params->dim_z);
				grid_points[i].neighborhood[idx] = nb_id;
				grid_points[i].neighbor_count += 1;
				idx++;
			}
		}
		//tcell data
		grid_points[i].binding_period = -1;
		grid_points[i].tissue_time_steps = -1;
		grid_points[i].id = -1;
		grid_points[i].from_id = -1;
		grid_points[i].num_bound = 0;
		for(int j = 0; j < 27; j++){
			grid_points[i].bound_from[j] = -1;
		}
	}
	state[tid] = local_state;
}

//curand kernels
__global__ void setup_curand(unsigned long long seed,
	unsigned long long offset, curandState* state){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void device_generate_tcells(Options* params, GridPoint* grid_points, 
											int* num_circulating_tcells){
	atomicAdd(num_circulating_tcells, params->tcell_generation_rate);
	if(*num_circulating_tcells < 0) *num_circulating_tcells = 0;
	// printf("num_circulating_tcells = %d\n",*num_circulating_tcells);
}

__global__ void device_update_circulating_tcells(Options* params,
												int* num_circulating_tcells,
												int* tcells_generated,
												GridPoint* grid_points,
												curandState* state,
												int num_xtravasing) {

	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_xtravasing;
	int step = blockDim.x*gridDim.x;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState local_state = state[tid];

	for(int i = start; i < maximum; i += step) {
		int x = get_between(&local_state, 0, params->dim_x-1);
		int y = get_between(&local_state, 0, params->dim_y-1);
		int z = get_between(&local_state, 0, params->dim_z-1);
		int id = device_to_1d(x,y,z,params->dim_x, params->dim_y, params->dim_z);
		int life_time = curand_poisson(&local_state, params->tcell_tissue_period);
		bool success = device_add_tcell(params,
							grid_points,
							tcells_generated,
							id, life_time);
  		if(success) {
  			// printf("### Added tcell to %d,%d,%d\n",x,y,z);
  			atomicSub(num_circulating_tcells, -1);
  		} else {
  			// printf("### Couldn't add tcell to %d, %d,%d\n", x, y, z);
  		}
	}
	state[tid] = local_state;
}

/**
* Begin T Cell Kernels
**/
__global__ void device_age_tcells(Options* params, GridPoint* grid_points){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
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
		if(grid_points[i].num_bound == 0){
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
	state[tid] = local_state;
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
/**
* End T Cell Kernels
**/

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
		grid_points[i].nb_chemokine = 0.0f;
	}
}

__global__ void device_update_virions(Options* params, GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	for(int i = start; i < maximum; i += step) {
		float v = grid_points[i].virions;
		float nb_v = grid_points[i].nb_virions;
		int num_nb = grid_points[i].num_neighbors;

		float virions_diffused = v*params->virion_diffusion;
		float virions_left = v - virions_diffused;
		float avg_nb_virions = (virions_diffused + nb_v * params->virion_diffusion)/(num_nb + 1);

		grid_points[i].virions = virions_left + avg_nb_virions;
		grid_points[i].virions = (1.0 - params->virion_clearance)*grid_points[i].virions;
		if(grid_points[i].virions < 0.0) grid_points[i].virions = 0.0;
		grid_points[i].nb_virions = 0.0f;
	}
}

__global__ void device_accumulate(Options* params, GridPoint* grid_points, int num_adj,
	int* loop_x, int* loop_y, int* loop_z) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	for(int id = start; id < maximum; id += step) {
		grid_points[id].num_neighbors = 0;
		grid_points[id].nb_virions = 0.0f;
		for(int j = 0; j < num_adj; j++) {
			int zz = id / (params->dim_x * params->dim_y);
			int i = id % (params->dim_x * params->dim_y);
			int yy = i / params->dim_x;
			int xx = i % params->dim_x;

			xx += loop_x[j];
			yy += loop_y[j];
			zz += loop_z[j];

			if(check_bounds(xx,yy,zz,params)) {
				int id_nb = device_to_1d(xx,yy,zz,
								params->dim_x, params->dim_y, params->dim_z);
				grid_points[id].nb_virions += grid_points[id_nb].virions;
				grid_points[id].num_neighbors += 1;
			}
		}
	}
	for(int id = start; id < maximum; id += step) {
		grid_points[id].nb_chemokine = 0.0f;
		for(int j = 0; j < num_adj; j++) {
			int zz = id / (params->dim_x * params->dim_y);
			int i = id % (params->dim_x * params->dim_y);
			int yy = i / params->dim_x;
			int xx = i % params->dim_x;

			xx += loop_x[j];
			yy += loop_y[j];
			zz += loop_z[j];

			if(check_bounds(xx,yy,zz,params)) {
				int id_nb = device_to_1d(xx,yy,zz,
								params->dim_x, params->dim_y, params->dim_z);
				grid_points[id].nb_chemokine += grid_points[id_nb].chemokine;
			}
		}
	}
}

__global__ void device_update_epicells(Options* params, GridPoint* grid_points, curandState* state){
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curandState local_state = state[tid];
	for(int i = start; i < maximum; i += step) {
		bool produce_virions = false;
		switch(grid_points[i].epicell_status) {
			case 0:
				if(grid_points[i].virions > 0) {
					if(trial_success(&local_state, params->infectivity*grid_points[i].virions)) {
						grid_points[i].epicell_status = 1;
					}
				}
				break;
			case 1:
				grid_points[i].incubation_time_steps--;
				if(grid_points[i].incubation_time_steps <= 0) {
					grid_points[i].epicell_status = 2;
				}
				break;
			case 2:
				grid_points[i].expressing_time_steps--;
				if(grid_points[i].expressing_time_steps <= 0) {
					grid_points[i].epicell_status = 4;
				} else {
					produce_virions = true;
				}
				break;
			case 3:
				grid_points[i].apoptotic_time_steps--;
				if(grid_points[i].apoptotic_time_steps <= 0) {
					grid_points[i].epicell_status = 4;
				} else if (grid_points[i].incubation_time_steps==0) {
					produce_virions = true;
				}
				break;
			default: break;
		}
		if(produce_virions) {
			grid_points[i].virions += params->virion_production;
			if((grid_points[i].chemokine + params->chemokine_production) < 1.0) {
				grid_points[i].chemokine = (grid_points[i].chemokine + params->chemokine_production);
			} else {
				grid_points[i].chemokine = 1.0;
			}
		}
	}
	state[tid] = local_state;
}

__global__ void device_initialize_infection(int x,int y,int z,float amount,
											GridPoint* grid_points, Options* params){
	int id = device_to_1d(x,y,z,
								params->dim_x, params->dim_y, params->dim_z);
	grid_points[id].virions = amount;
}

__global__ void device_reduce_values(Options* params, GridPoint* grid_points,
									uint64_cu* total_virions,
									uint64_cu* total_tcells,
									uint64_cu* total_healthy,
									uint64_cu* total_incubating,
									uint64_cu* total_expressing,
									uint64_cu* total_apoptotic,
									uint64_cu* total_dead,
									float* total_chem) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = params->num_points;
	int step = blockDim.x*gridDim.x;
	// NOTE: These adds might need to be atomic!
	for(int i = start; i < maximum; i+=step) {
		GridPoint gp = grid_points[i];
		atomicAdd(total_virions, (uint64_cu)gp.virions);
		atomicAdd(total_chem, (float)gp.chemokine);
		if(gp.id != -1){
			atomicAdd(total_tcells, 1);
		}
		switch(gp.epicell_status) {
			case 0:
				atomicAdd(total_healthy, 1);
				break;
			case 1:
				atomicAdd(total_incubating, 1);
				break;
			case 2:
				atomicAdd(total_expressing, 1);
				break;
			case 3:
				atomicAdd(total_apoptotic, 1);
				break;
			case 4:
				atomicAdd(total_dead, 1);
			default: break;
		}
	}
}

__global__ void setup_stencil(int* loop_x, int* loop_y, int* loop_z){
	loop_x[0] = -1;
	loop_x[1] = -1;
	loop_x[2] = -1;
	loop_x[3] = -1;
	loop_x[4] = -1;
	loop_x[5] = -1;
	loop_x[6] = -1;
	loop_x[7] = -1;
	loop_x[8] = -1;
	loop_x[9] = 0;
	loop_x[10] = 0;
	loop_x[11] = 0;
	loop_x[12] = 0;
	loop_x[13] = 0;
	loop_x[14] = 0;
	loop_x[15] = 0;
	loop_x[16] = 0;
	loop_x[17] = 1;
	loop_x[18] = 1;
	loop_x[19] = 1;
	loop_x[20] = 1;
	loop_x[21] = 1;
	loop_x[22] = 1;
	loop_x[23] = 1;
	loop_x[24] = 1;
	loop_x[25] = 1;
	loop_y[0] = -1;
	loop_y[1] = -1;
	loop_y[2] = -1;
	loop_y[3] = 0;
	loop_y[4] = 0;
	loop_y[5] = 0;
	loop_y[6] = 1;
	loop_y[7] = 1;
	loop_y[8] = 1;
	loop_y[9] = -1;
	loop_y[10] = -1;
	loop_y[11] = -1;
	loop_y[12] = 0;
	loop_y[13] = 0;
	loop_y[14] = 1;
	loop_y[15] = 1;
	loop_y[16] = 1;
	loop_y[17] = -1;
	loop_y[18] = -1;
	loop_y[19] = -1;
	loop_y[20] = 0;
	loop_y[21] = 0;
	loop_y[22] = 0;
	loop_y[23] = 1;
	loop_y[24] = 1;
	loop_y[25] = 1;
	loop_z[0] = -1;
	loop_z[1] = 0;
	loop_z[2] = 1;
	loop_z[3] = -1;
	loop_z[4] = 0;
	loop_z[5] = 1;
	loop_z[6] = -1;
	loop_z[7] = 0;
	loop_z[8] = 1;
	loop_z[9] = -1;
	loop_z[10] = 0;
	loop_z[11] = 1;
	loop_z[12] = -1;
	loop_z[13] = 1;
	loop_z[14] = -1;
	loop_z[15] = 0;
	loop_z[16] = 1;
	loop_z[17] = -1;
	loop_z[18] = 0;
	loop_z[19] = 1;
	loop_z[20] = -1;
	loop_z[21] = 0;
	loop_z[22] = 1;
	loop_z[23] = -1;
	loop_z[24] = 0;
	loop_z[25] = 1;

}
