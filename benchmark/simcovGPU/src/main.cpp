#include "tissue.hpp"
#include "options.hpp"

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

int main(int argc, char** argv) {
	//Handle parameters
	Options opt = parse_args(argc, argv);

	//params for testing
	opt.num_points = opt.dim_x*opt.dim_y*opt.dim_z;
	if(opt.seed == -1){
		opt.seed = time(NULL);
	}

	Tissue* tissue = new Tissue(opt);

	int inf_gap_x = (int)opt.dim_x/(opt.num_infections+1);
	int inf_gap_y = (int)opt.dim_y/(opt.num_infections+1);
	int inf_gap_z = (int)opt.dim_z/(opt.num_infections+1);

	for(int i = 0; i < opt.num_infections; i++){
		tissue->initialize_infection(inf_gap_x + i*inf_gap_x,
										inf_gap_y + i*inf_gap_y,
										inf_gap_z + i*inf_gap_z,
										opt.initial_infection);
	}
	//run sim
	printf("iter,virs,tcell_vasc,tcell_tissue,healthy,incb,expr,apop,dead,chem\n");
	fprintf(tissue->fout,"iter virs tcell_vasc tcell_tissue healthy incb expr apop dead chem\n");
	// fprintf(tissue->fout,"iter virs tcell_vasc tcell_tissue healthy incb expr dead chem\n");
	// printf("iter,i,x,y,z,virs,chem,tcell_time,epicell_status\n");
	cudaProfilerStart();
	for(int iter = 0; iter < opt.timesteps; iter++) {
		if(opt.sample_period > 0){
			if(iter%opt.sample_period == 0) {
				tissue->reduce_values();
				tissue->print_stats(iter);
				// tissue->dump_state(iter);
			}
		}
		if(iter > opt.tcell_initial_delay) {
			tissue->generate_tcells();
		}
		tissue->update_circulating_tcells();
		tissue->update_epicells();
		tissue->update_tissue_tcells();
		tissue->update_chemokine();
		tissue->update_virions();
	}
	cudaProfilerStop();

	delete tissue;
	return 0;
}