sudo nvidia-smi -ac "715,1328"
gevo-evolve -P profile.json -fitf kernel_time --mutop c,i,p --pop_size 256 --cxpb 0.7 --mupb 0.3 --random_seed 3 --err_rate "3.0s" | tee -i progress

