#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import extract_2D_tuning, extract_1D_tuning
import matplotlib.pyplot as plt

#%% Load YAML file
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Open-field
path = '../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621'
data = load_data(path)

#%% Pre-process data
data=preprocess_data(data,params)

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
