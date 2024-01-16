#%%
%load_ext autoreload
%autoreload 2

#%%
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.decoding import bayesian_decode
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
#plt.style.use('plot_style.mplstyle')
#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load data
data = preprocess_data(load_data('../' + params['path_to_dataset']+'/CA1/M246/M246_OF_1'), params)

#%% Select neurons
# Selected_neurons should contain the indices of the neurons used to decode behavior
# It is essential that the number of neurons used is the same across compared sessions
selected_neurons = np.random.choice(data['binaryData'].shape[1], 32)

#%% Establish training/testing frames
trainingFrames = np.zeros(len(data['caTime']), dtype=bool)
trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
testingFrames = ~trainingFrames

# Exclude immobility periods from both sets
trainingFrames[~data['running_ts']] = False
testingFrames[~data['running_ts']] = False

#%%
# mask = np.zeros(data['rawData'].shape[1], dtype=bool)
# mask[selected_neurons] = True
# frameMask = np.where(trainingFrames)
#%%
trainingData = data['rawData'][trainingFrames]
test = trainingData[:,selected_neurons]
test.shape


#%% Decode position in the open field
decoding_score, decoding_zscore, decoding_pvalue, decoding_error, shuffled_error, prediction = bayesian_decode(
    data['position'][:,0],
    data['binaryData'],
    params,
    selected_neurons,
    trainingFrames,
    testingFrames)
    
    
# %%
