#%% Imports
import yaml
import os
import h5py

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
for sessions in sessionsList:
    [ca_trace, ca_time, ]


# Load each matfile

# Binarize calcium traces

# Interpolate behavior

# Compute velocities

# Find periods of immobility, create an inclusion vector

# Compute occupancy and joint probabilities

# Compute tuning curves for every cell
# [MI, PDF, occupancy_vector, prob_being_active, tuning_curve ] = extract_1D_information(binarized_trace, interp_behav_vec, bin_vector, inclusion_vector)

# Smooth

# Shuffle, compute significance

# Compute split-half stability

# Decoding (jacknife? Decoding using 1,2,4,8,16,32,64,128,256,512, ... max cells)

# Save the data in a neat format!

#%% Group analysis
# Plot stuff!

