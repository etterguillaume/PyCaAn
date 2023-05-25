#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import numpy as np
from seqnmf import seqnmf
import yaml
import os
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Open parameters
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Import data
session_path = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
#session_path = '../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
data = load_data(session_path)
data = preprocess_data(data, params)

#%%
seqLength = params['sequence_length']*params['sampling_frequency']
running_data = data['binaryData'][data['running_ts']].T

#%% Extract sequences #TODO serialize
W, H, cost, loadings, power = seqnmf.seqnmf(running_data, K=2, L=10, Lambda=0.001)
# %%
plt.plot(H[0])
#plt.xlim([2780,2800])
#%%
H.shape

# %%
plt.imshow(W[:,1,:])
# %%
run_elapsed_time=data['elapsed_time'][data['running_ts']]
run_distance_travelled=data['distance_travelled'][data['running_ts']]
run_prospective_time=data['time2stop'][data['running_ts']]
run_prospective_distance=data['distance2stop'][data['running_ts']]
# %%
plt.scatter(run_distance_travelled, run_prospective_distance, c=H[1])
# %%
# Can we decode location, tine, dist.... using sequences??

