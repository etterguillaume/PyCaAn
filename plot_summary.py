#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('plot_style.mplstyle')
from functions.data_processing.dataloaders import load_data
from functions.signal_processing.signal_processing import binarize_ca_traces, interpolate_behavior, compute_velocity
#%%
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Establish list of animals
foldersList = os.listdir(params['path_to_dataset'] + os.sep + 'open_field')
for i, folder in enumerate(foldersList):
  if not folder.startswith('M'):
    foldersList.pop(i)
print(f'Found {len(foldersList)} subjects')

#%% Establish list of sessions
sessionsList = []
for folder in foldersList:
    sessions = os.listdir(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + folder)
    for session in sessions:
        if os.path.isfile(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + folder + os.sep + session + os.sep + 'ms.mat'):
            sessionsList.append(folder + os.sep + session)
print(f'Found {len(sessionsList)} total sessions')

#%%TODO: fix path issues
#data = load_data(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + session)
data = load_data('/Users/guillaumeetter/Documents/datasets/calcium_imaging/open_field/M246/J20_246_lego005_20180621/')

#%%
plt.figure(figsize=(3,3))
plt.subplot(221)
plt.plot(data['position'][0],data['position'][1]) # Plot trajectory
plt.title('Location (cm)')
plt.xlim([0,params['open_field_width']])
plt.ylim([0,params['open_field_width']])
plt.xticks([0,25,50])
plt.yticks([0,25,50])

plt.subplot(222)
plt.imshow(data['corrProj'],aspect='auto',cmap='bone')
#plt.imshow(np.max(data['SFPs'],axis=0),aspect='auto',cmap='bone')
plt.axis('off')

plt.subplot(212)
max_val=np.max(data['caTrace'])
for i in range(params['max_ca_traces_to_plot']):
    plt.plot(data['caTime'][0,:],data['caTrace'][i,:]*params['plot_gain']+max_val*i/params['plot_gain'], c=(1-i/50,.6,i/50))

# %%
data['numNeurons']
# %%
plt.plot(np.cos(np.arange(100)))

# %%
data['SFPs'].shape
# %%
data['numNeurons']
# %%
