#%%
%load_ext autoreload
%autoreload 2

#%%
from functions.dataloaders import load_data
from functions.signal_processing import extract_tone, preprocess_data
import yaml
import matplotlib.pyplot as plt

#%%
with open('params.yaml','r') as file:
    params = yaml.full_load(file)
#%%
#path = '../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
#path='/Users/guillaumeetter/Documents/datasets/calcium_imaging/CA1/M991/M991_legoSeqLT_20190313'
#path = '../../datasets/calcium_imaging/CA1/M989/M989_legoSeqLT_20190313'
path = '../../datasets/calcium_imaging/CA1/M990/M990_legoSeqLT_8Hz_20190320'

#%%
data=load_data(path)

#%%
data = preprocess_data(data, params)

#%%
data = extract_tone(data, params)

#%%
plt.plot(data['seqLT_state'])

# %%
import matplotlib
plt.figure()
cmap = matplotlib.cm.get_cmap('magma')
plt.plot(data['position'][:,0],color=cmap(data['seqLT_state']/4))
