#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Import dependencies
import yaml
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.data_embedding import train_embedding_model
from functions.decoding import train_linear_decoder
from functions.datasets import generateDataset, split_to_loaders

from functions.analysis import analyze_AE_reconstruction, analyze_decoding

import numpy as np
from scipy.stats import pearsonr as corr
from functions.plotting import plot_losses


#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '../../datasets/calcium_imaging/M246/M246_LT_7'
data = load_data(session_path)

#%% Preprocessing 
data = preprocess_data(data,params)
data['caTrace']=data['caTrace'][:,0:params['input_neurons']]

#%%
dataset = generateDataset(data)

#%% Split dataset into 
train_loader, test_loader = split_to_loaders(dataset, params)

#%%
embedding_model, train_loss, test_loss = train_embedding_model(params, train_loader, test_loader)
plot_losses(train_loss, test_loss, title='Embedding model')

#%% Train decoder on embedding and location
embedding_decoder, train_loss, test_loss = train_linear_decoder(params, embedding_model, train_loader, test_loader)
plot_losses(train_loss, test_loss, title='Decoder')
# %% Compute 
train_accuracy, train_precision, train_recall, train_F1 = analyze_AE_reconstruction(params, embedding_model, train_loader)
test_accuracy, test_precision, test_recall, test_F1 = analyze_AE_reconstruction(params, embedding_model, test_loader)

# %% Compute decoding errors
train_decoding_error, train_decoder_stats = analyze_decoding(params, embedding_model, embedding_decoder, train_loader)
test_decoding_error, test_decoder_stats = analyze_decoding(params, embedding_model, embedding_decoder, test_loader)

# %% Visualize reconstruction
