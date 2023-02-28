##%
# This file is to extract each session individually

#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import extract_tuning_curves
import matplotlib.pyplot as plt
import h5py

#%% Load YAML file
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% List all sessions here

#%% Load data
path = '../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621'
data = load_data(path)

#%% Check if data already analyzed
# TODO if folder don't exist, create it


#%% Pre-process data
data=preprocess_data(data,params)
#TODO extract basic stats, num cells, distance travelled, etc in info.h5



#%% Extract tuning curves
# Always check if data exists before analyzing!!
# save tuning_curves and in individual files (e.g location.h5, velocity.h5,... which contains tuning_curve, AMI, occupancy, etc for each neuron)

# For all:
# extract elapsed time tuning
# extract distance travelled tuning
# extract velocity tuning

# for all, maze-size dependent:
# extract location

# for all, maze-type dependent (LT, non-LT)
# extract LR direction tuning OR extract HD tuning

# for toneLT, seqLT
# extract tone tuning

# for seqLT
# extract tone type tuning


# TODO check if folder exists, check if params['overwrite'] is 'changes_only', 'always' or 'never'
# TODO if exists, check results contents. If params_old==params_new, decide whether overwrite

#%% Extract tuning correlatins

#%% Decode (OPTIONAL)

#%% POOLED DATA ANALYSIS
#TODO create new folder with pooled results

#%% Test bimodality for mutual information, activity, stability etc etc



#%% Chronic alignment (across days)
# Using SFPs+ corrProj, pnrProj, meanProj; output identity matrix as dictionary?
#TODO create new folder with alignment results

#%% UMAP hyperalignment (across days, mice, regions)
#TODO create new folder with hyperalignment results


