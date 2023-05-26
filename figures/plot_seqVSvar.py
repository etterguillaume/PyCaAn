#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import numpy as np
from seqnmf import seqnmf
import yaml
import os
from scipy.signal import find_peaks
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Open parameters
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Import data
session_path = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
#session_path = '../../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
data = load_data(session_path)
data = preprocess_data(data, params)

#%%
seqLength = params['sequence_length']*params['sampling_frequency']
running_data = data['binaryData'][data['running_ts']].T

#%% Extract sequences #TODO serialize
W, H, cost, loadings, power = seqnmf(running_data,
                                     K=2,
                                     L=seqLength,
                                     Lambda=0.001,
                                     max_iter=10)
# %%
plt.plot(H[0])
#plt.xlim([2780,2800])

# %%
plt.imshow(W[:,0,:])
# %%
run_location=data['position'][data['running_ts'],0]
run_velocity=data['velocity'][data['running_ts']]
run_elapsed_time=data['elapsed_time'][data['running_ts']]
run_distance_travelled=data['distance_travelled'][data['running_ts']]
run_prospective_time=data['time2stop'][data['running_ts']]
run_prospective_distance=data['distance2stop'][data['running_ts']]
# %% 
# Note that H only gives the *location* of sequences, note some measure of 'sequenceness'
# One has to look before/after each peak of H to retrieve what happened in neural activity
input_signal = H[0]
peaks, _ = find_peaks(input_signal, distance=seqLength*2, threshold=np.max(input_signal)*.1)

plt.plot(H[0])
plt.plot(peaks, H[0,peaks], "x",color='C6')

#%%
windows=[]
for i, peak in enumerate(peaks):
    if int(peak+seqLength/2)<len(input_signal) and int(peak-seqLength/2)>0:
        windows.append([int(peak-seqLength/2), int(peak+seqLength/2)])

windows=np.array(windows)

#%%
for i, window in enumerate(windows):
    plt.plot(run_location[window[0]:window[1]],color='C0')

#%%
for i, window in enumerate(windows):
    plt.plot(run_velocity[window[0]:window[1]],color='C1')

#%%
for i, window in enumerate(windows):
    plt.plot(run_elapsed_time[window[0]:window[1]],color='C2')

#%%
for i, window in enumerate(windows):
    plt.plot(run_distance_travelled[window[0]:window[1]],color='C3')
        
#%%
for i, window in enumerate(windows):
    plt.plot(run_prospective_distance[window[0]:window[1]],color='C4')

# %%
# Can we decode location, tine, dist.... using sequences??

