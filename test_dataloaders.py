#%%
from functions.dataloaders import load_data
from functions.signal_processing import extract_tone, preprocess_data
import yaml
import matplotlib.pyplot as plt

#%%
with open('params.yaml','r') as file:
    params = yaml.full_load(file)
#%%
#path = '../../datasets/calcium_imaging/M986/M986_legoSeqLT_20190312'
path='/Users/guillaumeetter/Documents/datasets/calcium_imaging/CA1/M991/M991_legoSeqLT_20190313'
#%%
data=load_data(path)

#%%
data = preprocess_data(data, params)
# %%
assert data is not None

#%%
data = extract_tone(data, params)
# %%
import matplotlib
plt.figure()
cmap = matplotlib.cm.get_cmap('magma')
plt.plot(data['position'][:,0],color=cmap(data['seqLT_state']/4))

# %%