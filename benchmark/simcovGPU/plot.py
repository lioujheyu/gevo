#!/usr/bin/env python3

#%%
import matplotlib
import csv
import pandas as pd
import subprocess

#%%
run = []
count = 10
for i in range(count):

    subprocess.run(
        [ './simcov_gpu.out',
          '--dim_x', '100',
          '--dim_y', '100',
          '--sample_period', '1',
          '--tcell_initial_delay', '500',
          '--timesteps', '2500',
          '--seed', '7'
          ],
        stdout=subprocess.DEVNULL )
    with open('output', 'r') as dataf:
        df = pd.read_csv(dataf, delimiter=' ', index_col='iter')

    run.append(df)
#%%
c_label = 'healthy'
ds = pd.DataFrame(run[0][c_label])
for i in range(1,count):
    ds = pd.concat([ds, pd.DataFrame(run[i][c_label])], axis=1)

ds.plot(title=c_label, legend=False,)    
# %%
