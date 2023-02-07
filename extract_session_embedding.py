#%% Import dependencies
import torch
from torchvision import transforms
import yaml
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.data_embedding import train_embedding_model
from functions.decoding import train_linear_decoder
from functions.datasets import generateDataset, split_to_loaders

import numpy as np
from scipy.stats import pearsonr as corr

import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

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

#%% Train decoder on embedding and location
embedding_decoder = train_linear_decoder(params, embedding_model, train_loader, test_loader)



# %%
