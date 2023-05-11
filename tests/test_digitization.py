#%%
%load_ext autoreload
%autoreload 2

#%%
from functions.dataloaders import load_data
import os
from functions.signal_processing import preprocess_data
from functions.tuning import assess_covariate
import yaml
import numpy as np
import matplotlib.pyplot as plt
#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

# #%% Load folders to analyze from yaml file?
# with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
#     session_file = yaml.full_load(file)
# session_list = session_file['sessions']
# path = session_list[232]
#%%
path = '../../../datasets/calcium_imaging/CA1/M1087/M1087_legoOF_20191119'

#%%
data = load_data(path)
data = preprocess_data(data, params)

#%%
from functions.tuning import extract_tuning
#%%
bin_vec=(np.arange(0,50+params['spatialBinSize'],params['spatialBinSize']),
        np.arange(0,50+params['spatialBinSize'],params['spatialBinSize']))
AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_tuning(data['binaryData'],data['position'],data['running_ts'],bins=bin_vec)

#%% Assertions




# %%
