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

#%% Extract tuning correlatins


#%% Decode


##%
# This file is to extract information across session
# Define across what (days, vs regions, mice etc)

#%% POOLED DATA ANALYSIS
#TODO create new folder with pooled results

#%% Test bimodality for mutual information, activity, stability etc etc



#%% Chronic alignment (across days)
# Using SFPs+ corrProj, pnrProj, meanProj; output identity matrix as dictionary?
#TODO create new folder with alignment results

#%% UMAP hyperalignment (across days, mice, regions)
#TODO create new folder with hyperalignment results


