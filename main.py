#%% Imports
import yaml
import os
import pandas
from scipy.io import loadmat

#%% Load YAML file
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Establish list of animals
# Use pandas to create table of all recordings, input their date (use as ID). Row = mouse, column = experiment/session
# Save in results, use this as list of sessions to process

#%% Subject analysis
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

