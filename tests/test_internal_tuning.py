#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
import h5py
import numpy as np
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.tuning import extract_tuning
import matplotlib.pyplot as plt
from scipy.stats import zscore

#%% Load YAML file
with open('../params_CA1.yaml','r') as file:
    params = yaml.full_load(file)

#%% Linear track
path = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
data = load_data(path)
data = preprocess_data(data,params)

#%% Load corresponding embedding
working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )

# Load embedding
try:
    embedding_file = h5py.File(os.path.join('..',working_directory,'embedding.h5'),'r')
except:
    print('Could not find embedding file. Please first extract embedding data.')
# embedding = embedding_file['embedding'][()]
train_embedding = embedding_file['train_embedding'][()]
trainingFrames = embedding_file['trainingFrames'][()]

std_embedding = zscore(train_embedding)

#%%
#TODO parametrize dimensionality
bin_vec = (np.arange(-params['max_internal_distance'],params['max_internal_distance']+params['internalBinSize'],params['internalBinSize']),
           np.arange(-params['max_internal_distance'],params['max_internal_distance']+params['internalBinSize'],params['internalBinSize']),
           np.arange(-params['max_internal_distance'],params['max_internal_distance']+params['internalBinSize'],params['internalBinSize']),
           np.arange(-params['max_internal_distance'],params['max_internal_distance']+params['internalBinSize'],params['internalBinSize']))

(
        info,
        p_value,
        occupancy_frames,
        active_frames_in_bin,
        tuning_curves,
        peak_loc,
        peak_val,
    ) = extract_tuning(
            data['binaryData'][trainingFrames],
            std_embedding,
            np.ones(sum(trainingFrames),dtype='bool'), # Immobility already filtered
            bins=bin_vec
                                        )

# %%
