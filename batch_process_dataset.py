#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import binarize_ca_traces, interpolate_behavior, compute_velocity
from functions.tuning import extract_2D_tuning

#%% Load YAML file
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

#%% Subject analysis
#for session in tqdm.tqdm(sessionsList):
# Check if data exists
data = load_data(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + session)
