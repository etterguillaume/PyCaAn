#%% Imports
import yaml
import os
from functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from argparse import ArgumentParser
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
for session in sessionsList:
    if os.path.exists(# Check if session exists
        data = load_data(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + session)

#data = load_data('/Users/guillaumeetter/Documents/datasets/calcium_imaging/open_field/M246/J20_246_lego005_20180621/')

#%%
name = 'M246'
plot_summary(data, params, name, plot=True)
# %%
data.keys()

# %%
data['experiment']

# %% Save figure