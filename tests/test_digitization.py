#%%
%load_ext autoreload
%autoreload 2

#%%
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import assess_covariate
import yaml
import numpy as np
from numpy import digitize
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
#%%
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
    session_file = yaml.full_load(file)
session_list = session_file['sessions']
path = session_list[232]
#%%
path = '../../datasets/calcium_imaging/CA1/M246/M246_OF_1'

#%%
data=load_data(path)
data = preprocess_data(data, params)

#%%
binaryData=data['binaryData']
inclusion_ts=data['running_ts']
interpolated_var=data['position']

var_length=50
bin_size=4

#%%
X_bin_vector = np.arange(0,var_length+bin_size,bin_size)
Y_bin_vector = np.arange(0,var_length+bin_size,bin_size)
binaryData = binaryData[inclusion_ts]
interpolated_var = interpolated_var[inclusion_ts]
numFrames, numNeurons = binaryData.shape
occupancy_frames = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1), dtype=int)

# Compute occupancy
bin_vector = np.zeros(numFrames, dtype=int) # Vector that will specificy the bin# for each frame
ct=0
for y in range(len(Y_bin_vector)-1):
    for x in range(len(X_bin_vector)-1):
        frames_in_bin = (interpolated_var[:,0] >= X_bin_vector[x]) & (interpolated_var[:,0] < X_bin_vector[x+1]) & (interpolated_var[:,1] >= Y_bin_vector[y]) & (interpolated_var[:,1] < Y_bin_vector[y+1])
        occupancy_frames[y,x] = np.sum(frames_in_bin) # How many frames for that bin
        bin_vector[frames_in_bin] = ct
        ct+=1

#%% Digitize


#%% 
#assert


