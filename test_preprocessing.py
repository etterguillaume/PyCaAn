#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2

#%% Import dependencies
import yaml
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
from functions.dataloaders import load_data
from functions.signal_processing import compute_velocity, compute_distance_time, interpolate_2D

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
#session_path = '../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
data = load_data(session_path)

#%% Preprocessing 
data['position'] = interpolate_2D(data['position'], data['behavTime'], data['caTime'])
data['velocity'], data['running_ts'] = compute_velocity(data['position'], data['caTime'], params['speed_threshold'])

#%%
elapsed_time, traveled_distance = compute_distance_time(data['position'], data['velocity'], data['caTime'], 2)