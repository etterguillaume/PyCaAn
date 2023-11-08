#%%
%load_ext autoreload
%autoreload 2

#%% Import dependencies
import yaml

from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import compute_distance_from_center, compute_velocity, compute_distance_time, interpolate_2D, preprocess_data

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%% Load parameters
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
#session_path = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
session_path = '../../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
#session_path = '../../../datasets/calcium_imaging/CA1/M1117/M1117_OF_20200131'
data = load_data(session_path)

#%%
data = preprocess_data(data, params)

#%%
dist_from_center = compute_distance_from_center(data['position'],45)

#%%
plt.scatter(data['position'][:,0],
data['position'][:,1],
c=dist_from_center
)

#%%
plt.plot(data['distance_travelled'])
plt.plot(data['distance2stop'])
#plt.plot(data['velocity'])
plt.xlim([2500,9500])

#%%
plt.scatter(data['distance_travelled'], data['distance2stop'],s=.1)


#%% Preprocessing 
data['position'] = interpolate_2D(data['position'], data['behavTime'], data['caTime'])
data['velocity'], data['running_ts'] = compute_velocity(data['position'], data['caTime'], params['speed_threshold'])

#%%
elapsed_time, traveled_distance = compute_distance_time(data['position'], data['velocity'], data['caTime'], 2)