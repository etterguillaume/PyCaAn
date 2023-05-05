#%%
%load_ext autoreload
%autoreload 2

#%%
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import assess_covariate
import yaml
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('plot_style.mplstyle')
#%%
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
    session_file = yaml.full_load(file)
#%%
path = '../../datasets/calcium_imaging/CA1/M246/M246_OF_1'

#%%
data=load_data(path)
data = preprocess_data(data, params)

# %%
info,pvalue = assess_covariate(data['position'],
                               data['heading'],
                               data['running_ts'],
                               100,
                               params['spatialBinSize'],
                               360,
                               9)
print(f'location x HD: {info}, pvalue: {pvalue}')
# %%
info,pvalue = assess_covariate(data['position'][:,0],
                               data['elapsed_time'],
                               data['running_ts'],
                               100,
                               params['spatialBinSize'],
                               params['max_temporal_length'],
                               params['temporalBinSize'])
print(f'location x time info: {info}, pvalue: {pvalue}')

# %%
info,pvalue = assess_covariate(data['velocity'],
                               data['elapsed_time'],
                               data['running_ts'],
                               params['max_velocity_length'],
                               params['velocityBinSize'],
                               params['max_temporal_length'],
                               params['temporalBinSize'])
print(f'velocity x time info: {info}, pvalue: {pvalue}')
# %%
