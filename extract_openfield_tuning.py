#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
import tqdm
from functions.data_processing.dataloaders import load_data
from functions.signal_processing.signal_processing import binarize_ca_traces, interpolate_behavior, compute_velocity
from functions.tuning_curves.tuning import extract_2D_tuning

#%% Load YAML file
print('Opening parameters file... ', end='')
with open('params.yaml','r') as file:
    params = yaml.full_load(file)
print('Done!')
#%% Establish list of animals
print('Listing subjects... ', end='')
foldersList = os.listdir(params['path_to_dataset'] + os.sep + 'open_field')
for i, folder in enumerate(foldersList):
  if not folder.startswith('M'):
    foldersList.pop(i)
print('Done!')
print(f'Found {len(foldersList)} subjects')

#%% Establish list of sessions
print('Listing sessions... ', end='')
sessionsList = []
for folder in foldersList:
    sessions = os.listdir(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + folder)
    for session in sessions:
        if os.path.isfile(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + folder + os.sep + session + os.sep + 'ms.mat'):
            sessionsList.append(folder + os.sep + session)
print('Done!')
print(f'Found {len(sessionsList)} total sessions')

#%% Subject analysis
#for session in tqdm.tqdm(sessionsList):
# Check if data exists
data = load_data(params['path_to_dataset'] + os.sep + 'open_field' + os.sep + session)

# Apply thresholds_1 here (min duration/frames, clean data etc)

#for session in tqdm.tqdm(sessionsList):
binarized_traces = binarize_ca_traces(data['caTrace'], params['z_threshold'],params['sampling_frequency'])
interpolated_position = interpolate_behavior(data['position'], data['behavTime'], data['caTime'])
velocity, running_ts = compute_velocity(interpolated_position, data['caTime'], params['speed_threshold'])

# Apply thresholds_2 here (min num cells, coverage, distance travelled, speed etc etc)

#for session in tqdm.tqdm(sessionsList):
#%% Compute tuning curves for every cell
tuning_data = extract_2D_tuning(binarized_traces, interpolated_position, running_ts, params) # extract fields properties, stability, peak likelihood, dispersion, MI, MI sig, fields
# store all the data in a folder for large scale analysis

#%%
# Decoding (jacknife? Decoding using 1,2,4,8,16,32,64,128,256,512, ... max cells)

#%%
# Save the data in a neat format!
# Mice selected, number of cells per mouse, distance travelled, avg speed, etc etc


#%% Group analysis
# Test bimodality for mutual information, activity, stability etc etc
# Multidimensional clustering (k-means? tsne?)
# Per mouse analysis vs all mice pooled?

#%%
# Model w/ artificial agent
