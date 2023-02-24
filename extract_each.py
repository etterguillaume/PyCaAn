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

#%% Load YAML file
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%%
path = '../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621'
data = load_data(path)

#%% Check if data already analyzed
# TODO check if folder exists, check if params['overwrite']
# TODO if exists, check results contents. If params_old==params_new, decide whether overwrite
# TODO if folder don't exist, create it

# TODO save in results/output/folder


#%% Pre-process data
data=preprocess_data(data,params)

#%% Extract tuning curves
tuning_curves = extract_tuning_curves(data, params)
# save tuning_curves and in one file
#TODO extract fields

#%% Test bimodality for mutual information, activity, stability etc etc

#%% Extract tuning correlatins


#%% Decode


#%%
# Save the data in a neat format!
# TODO save region, age, etc etc



#%%
# Model w/ artificial agent
