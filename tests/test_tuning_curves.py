#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
import numpy as np
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data, extract_seqLT_tone, extract_tone
from pycaan.functions.tuning import extract_discrete_tuning, extract_tuning
import matplotlib.pyplot as plt

#%% Load YAML file
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Linear track
path = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
data = load_data(path)
data=preprocess_data(data,params)

#%%
bin_vec = (np.arange(0,params['max_temporal_length']+params['temporalBinSize'],params['temporalBinSize']))
info, p_value, occupancy_frames, active_frames_in_bin, tuning_curve, marginal_likelihood, peak_loc, peak_val = extract_tuning(
                                        data['binaryData'],
                                        data['elapsed_time'],
                                        data['running_ts'],
                                        bins=bin_vec)

#%% Interesting plot...
plt.hexbin(marginal_likelihood,peak_val, gridsize=15, C=info, cmap='Spectral_r'); plt.colorbar()

#%% LT direction test
info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, marginal_likelihood, peak_loc, peak_val = extract_discrete_tuning(
    data['binaryData'],
    data['LT_direction'],
    data['running_ts'],
    var_length=2
    )

#%% Interesting plot...
plt.hexbin(marginal_likelihood,peak_val, gridsize=15, C=info, cmap='Spectral_r'); plt.colorbar()


#%% Open-field
path = '../../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
data = load_data(path)
data=preprocess_data(data,params)

#%%
bin_vec=(np.arange(0,45+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,45+params['spatialBinSize'],params['spatialBinSize']))
info, p_value, occupancy_frames, active_frames_in_bin, tuning_curve, marginal_likelihood, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                data['position'],
                                                data['running_ts'],
                                                bins=bin_vec)

#%%
plt.hexbin(marginal_likelihood,peak_val, gridsize=30, C=info, cmap='Spectral_r'); plt.colorbar()

#%%
# data = extract_seqLT_tone(data,params)
data = extract_tone(data,params)

#%%
AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_discrete_tuning(data['binaryData'],
                                                data['binaryTone'],
                                                data['running_ts'],
                                                var_length=1,
                                                )

#%%
AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_discrete_tuning(data['binaryData'],
                                                data['seqLT_state'],
                                                data['running_ts'],
                                                var_length=3,
                                                )

#%%
binarized_trace = data['binaryData'][:,6]

#%%
plt.plot(binarized_trace)
#%%
plt.plot(data['position'][:,0],data['position'][:,1]); plt.axis('equal')
#%% Extract tuning curves
AMI, occupancy_frames, active_frames_in_bin, tuning_curve = extract_2D_tuning(binarized_trace, data['position'], data['running_ts'], 50, .5)
# %%
print(AMI)
#%%
plt.imshow(tuning_curve); plt.colorbar()
#%%
plt.imshow(active_frames_in_bin); plt.colorbar()
#%%
plt.imshow(occupancy_frames); plt.colorbar()
# %%



#%% Linear track
path = '../../datasets/calcium_imaging/CA1/M246/M246_LT_6'

data = load_data(path)
# %%
#%% Pre-process data
data=preprocess_data(data,params)

#%%
binarized_trace = data['binaryData'][:,18]
#%%
plt.figure()
plt.plot(binarized_trace)
plt.figure()
plt.plot(data['position'][:,0],data['position'][:,1]); plt.axis('equal')

# %%
AMI, occupancy_frames, active_frames_in_bin, tuning_curve = extract_1D_tuning(binarized_trace, data['position'][:,0], data['running_ts'], 100, 2.5)

print(AMI)
#%%
plt.plot(tuning_curve)
#%%
plt.plot(active_frames_in_bin)
#%%
plt.plot(occupancy_frames)

# %%
