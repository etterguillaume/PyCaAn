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
from numpy import digitize
import matplotlib.pyplot as plt
#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
    session_file = yaml.full_load(file)
session_list = session_file['sessions']
path = session_list[232]
#%%
path = '../../../datasets/calcium_imaging/CA1/M988/M988_legoOF_scrambled_20190116'

#%%
data = load_data(path)
data = preprocess_data(data, params)

#%%
from functions.tuning import extract_2D_tuning, extract_tuning

#%%
bins = (np.arange(0,45,4), np.arange(0,45,4))
AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_2D_tuning(data['binaryData'],data['position'],data['running_ts'],var_length=45,bin_size=4)
d_AMI, d_p_value, d_occupancy_frames, d_active_frames_in_bin, d_tuning_curve = extract_tuning(data['binaryData'],data['position'],data['running_ts'],bins)

#%% Assertions
#assert occupancy_frames == d_occupancy_frames
assert AMI.all()==d_AMI.all()
# assert tuning
# assert pval
# assert chi2
# assert active_in_frames




# %%
