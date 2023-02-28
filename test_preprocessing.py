#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Import dependencies
import yaml
#from .. import functions.dataloaders import load_data
from ..functions.dataloaders import load_data

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '../../datasets/calcium_imaging/M246/M246_LT_6'
data = load_data(session_path)

#%% Preprocessing 
params['data_type'] = 'binarized'
data = preprocess_data(data,params)
# %%
idx=6
plt.plot(data['caTrace'][:,idx])
plt.plot(data['procData'][:,idx])
plt.xlim([3800,4000])
# %%
