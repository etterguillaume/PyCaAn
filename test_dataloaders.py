#%%
%load_ext autoreload
%autoreload 2

#%%
from functions.dataloaders import load_data
from functions.signal_processing import extract_tone, preprocess_data, clean_timestamps
import yaml
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
#%%
with open('params.yaml','r') as file:
    params = yaml.full_load(file)
#%%
#path = '../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
path = '../../datasets/calcium_imaging/LS/M732/M732_LT_2018050701'
#path = '../../datasets/calcium_imaging/CA1/M246/M246_OF_1'
#path='/Users/guillaumeetter/Documents/datasets/calcium_imaging/CA1/M991/M991_legoSeqLT_20190313'
#path = '../../datasets/calcium_imaging/CA1/M989/M989_legoSeqLT_20190313'
#path = '../../datasets/calcium_imaging/CA1/M990/M990_legoSeqLT_8Hz_20190320'

#%%
data=load_data(path)

#%%
data = preprocess_data(data, params)

#%%
data = extract_tone(data, params)

# %%
