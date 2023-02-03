#%% Import dependencies
import torch
import yaml
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.model_training import train_embedder, train_decoder

from functions.datasets import generateDataset, split_to_loaders
from models.autoencoders import AE_MLP
from models.decoders import linear_decoder

import numpy as np
import umap.umap_ as umap

# Metrics
#BCE_error = torch.nn.BCELoss(reduction='none')

#TEMP
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
from functions.plotting import interactive_plot_manifold3D

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Specify device and seeds
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #TODO: define prefered device in params
device = torch.device('cpu')

torch.manual_seed(params['seed']) # Seed for reproducibility
np.random.seed(params['seed'])

#%% Load session
session_path = '../../datasets/calcium_imaging/M246/M246_LT_7'
data = load_data(session_path)

#%% Preprocessing 
data = preprocess_data(data,params)

# %% Establish model and objective functions
model = AE_MLP(input_dim=data['numNeurons'],latent_dim=64,output_dim=params['embedding_dims']).to(device) # TODO: add to params
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate']) # TODO: add to params
criterion = torch.nn.MSELoss()

#%%
dataset = generateDataset(data, params)

#%% Establish dataset
train_loader, test_loader = split_to_loaders(dataset, params)

#%% Train model
train_embedder(model, train_loader, test_loader, optimizer, criterion, device, params)

#%% Train decoder on embedding and location
embedding_decoder = linear_decoder(input_dims=params['embedding_dims'])

#%%
decoder_optimizer = torch.optim.AdamW(embedding_decoder.parameters(), lr=params['learning_rate'])

#%% Train decoder
for param in model.parameters(): # Freeze embedding model
    param.requires_grad = False # Freeze all weights

train_decoder(model, train_loader, test_loader, optimizer, criterion, device, params)


 # %%
