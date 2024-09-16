#pragma once
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <utility>

struct Options {
	int dim_x = 15000;
	int dim_y = 15000;
	int dim_z = 1;
	int num_infections = 1;
	long long whole_lung_x = 48000;
	long long whole_lung_y = 40000;
	long long whole_lung_z = 20000;
	int num_points = 15000*15000*1;
	int timesteps = 33120;
	int initial_infection = 1000;
	int incubation_period = 480;
	int apoptosis_period = 180;
	int expressing_period = 900;
	float infectivity = 0.001;
	float virion_production = 1.1;
	float virion_clearance = 0.004;
	float virion_diffusion = 0.15;
	float chemokine_production = 1.0;
	float chemokine_decay = 0.01;
	float chemokine_diffusion = 1.0;
	float min_chemokine = 1e-6;
	float antibody_factor = 1;
	int antibody_period = 5760;
	int tcell_generation_rate = 105000;
	int tcell_initial_delay = 10080;
	int tcell_vascular_period = 5760;
	int tcell_tissue_period = 1440;
	int tcell_binding_period = 10;
	float max_binding_prob = 1;
	bool tcells_follow_gradient = false;
	int seed = -1;
	int sample_period = 0;
	int sample_resolution = 1;
};

Options parse_args(int argc, char** argv);
int check_args(Options opt);