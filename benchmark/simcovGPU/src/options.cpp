#include "options.hpp"

Options parse_args(int argc, char** argv) {
	Options o;

	std::unordered_map<std::string, int*> argMapInts;
	std::unordered_map<std::string, long long*> argMapLongLongs;
	std::unordered_map<std::string, float*> argMapFloats;
	std::unordered_map<std::string, bool*> argMapBools;

	argMapInts.insert(std::make_pair("--dim_x", &o.dim_x));
	argMapInts.insert(std::make_pair("--dim_y", &o.dim_y));
	argMapInts.insert(std::make_pair("--dim_z", &o.dim_z));
	argMapInts.insert(std::make_pair("--timesteps", &o.timesteps));
	argMapInts.insert(std::make_pair("--num_infections", &o.num_infections));
	argMapInts.insert(std::make_pair("--incubation_period", &o.incubation_period));
	argMapInts.insert(std::make_pair("--apoptosis_period", &o.apoptosis_period));
	argMapInts.insert(std::make_pair("--expressing_period", &o.expressing_period));
	argMapInts.insert(std::make_pair("--tcell_generation_rate", &o.tcell_generation_rate));
	argMapInts.insert(std::make_pair("--tcell_initial_delay", &o.tcell_initial_delay));
	argMapInts.insert(std::make_pair("--tcell_vascular_period", &o.tcell_vascular_period));
	argMapInts.insert(std::make_pair("--tcell_tissue_period", &o.tcell_tissue_period));
	argMapInts.insert(std::make_pair("--tcell_binding_period", &o.tcell_binding_period));
	argMapInts.insert(std::make_pair("--sample_period", &o.sample_period));
	argMapInts.insert(std::make_pair("--seed", &o.seed));

	argMapFloats.insert(std::make_pair("--infectivity", &o.infectivity));
	argMapFloats.insert(std::make_pair("--virion_production", &o.virion_production));
	argMapFloats.insert(std::make_pair("--virion_clearance", &o.virion_clearance));
	argMapFloats.insert(std::make_pair("--virion_diffusion", &o.virion_diffusion));
	argMapFloats.insert(std::make_pair("--chemokine_production", &o.chemokine_production));
	argMapFloats.insert(std::make_pair("--virion_diffusion", &o.virion_diffusion));
	argMapFloats.insert(std::make_pair("--chemokine_production", &o.chemokine_production));
	argMapFloats.insert(std::make_pair("--chemokine_decay", &o.chemokine_decay));
	argMapFloats.insert(std::make_pair("--chemokine_diffusion", &o.chemokine_diffusion));
	argMapFloats.insert(std::make_pair("--min_chemokine", &o.min_chemokine));
	argMapFloats.insert(std::make_pair("--max_binding_prob", &o.max_binding_prob));

	for(int i = 1; i < argc; i++) {
		std::string argument = argv[i];
		std::string indicator = "--";

		if(argument.find(indicator) != std::string::npos) {
			if(argMapInts.find(argument) != argMapInts.end()){
				*(argMapInts.at(argument)) = atoi(argv[i+1]);
			} else if (argMapLongLongs.find(argument) != argMapLongLongs.end()) {
				*(argMapLongLongs.at(argument)) = atoll(argv[i+1]);
			} else if (argMapFloats.find(argument) != argMapFloats.end()) {
				*(argMapFloats.at(argument)) = atof(argv[i+1]);
			} else if (argMapBools.find(argument) != argMapBools.end()) {
				int val = atoi(argv[i+1]);
				if (val == 0) {
					*(argMapInts.at(argument)) = false;
				} else {
					*(argMapInts.at(argument)) = true;
				}
			}
		}
	}	

	return o;
}

int check_args(Options opt){
	int result = 1;
	return result;
}