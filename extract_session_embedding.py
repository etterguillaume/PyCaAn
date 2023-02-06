#%% Import dependencies
import torch
from torchvision import transforms
import yaml
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.data_embedding import train_embedder
from functions.decoding import train_decoder
from functions.datasets import generateDataset, split_to_loaders
from models.decoders import linear_decoder

import numpy as np
from scipy.stats import pearsonr as corr

# Metrics
#BCE_error = torch.nn.BCELoss(reduction='none')

#TEMP
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')
from functions.plotting import interactive_plot_manifold3D

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '../../datasets/calcium_imaging/M246/M246_LT_7'
data = load_data(session_path)

#%% Preprocessing 
data = preprocess_data(data,params)

#%%
dataset = generateDataset(data, params)

#%% Establish dataset
train_loader, test_loader = split_to_loaders(dataset, params)

embedding_model = train_embedding_model(params, train_loader, test_loader)

#%% Train decoder on embedding and location
embedding_decoder = train_linear_decoder(params, embedding_model, train_loader, test_loader)


#%% Validate models
# Validate reconstruction error, accuracy
# Validate decoding error, accuracy
total_inputs = np.empty((0,data['numNeurons']))
total_reconstructions = np.empty((0,data['numNeurons']))
total_predictions = np.empty(0)
total_positions = np.empty(0)
total_losses = np.empty(0)
total_pred_losses = np.empty(0)

for i, (x, position, _) in enumerate(test_loader):
    x = x.to(device)
    with torch.no_grad():
        reconstruction, embedding = model(x)
        loss = BCE_criterion(reconstruction, x) # Only compute loss on masked part
        pred = embedding_decoder(embedding)
        pred_loss = MSE_criterion(pred,position[:,0])

        total_inputs = np.append(total_inputs, x, axis=0)
        total_reconstructions = np.append(total_reconstructions, reconstruction, axis=0)
        total_losses = np.append(total_losses, loss)
        total_pred_losses = np.append(total_pred_losses, pred_loss)
        total_positions = np.append(total_positions, position[:,0], axis=0)
        total_predictions = np.append(total_predictions, pred[0], axis=0)

#%%
reconstruction_stats = corr(total_inputs.flatten(),total_reconstructions.flatten())
decoder_stats = corr(total_positions.flatten(),total_predictions.flatten())
avg_loss = np.mean(total_losses)
avg_pred_loss = np.mean(total_pred_losses)

#%%
plt.figure(figsize=(3,3))
plt.subplot(221)
x=total_inputs.flatten()
y=total_reconstructions.flatten()
a, b = np.polyfit(x, y, 1)
plt.scatter(x,y)
plt.plot(x, a*x+b,'r--')
plt.xlabel('Original input value')
plt.ylabel('Reconstructed input value')
plt.title(f'Reconstruction\nR2={reconstruction_stats[0].round(4)}, p={reconstruction_stats[1].round(4)}')

plt.subplot(222)
plt.hist(total_losses)
plt.xlabel('Loss')
plt.ylabel('Number')
plt.title(f'Embedder\nmean loss:{avg_loss.round(4)}')

plt.subplot(223)
x=total_positions[:,0].flatten()
y=total_predictions[:,0].flatten()
a, b = np.polyfit(x, y, 1)
plt.scatter(x,y)
plt.plot(x, a*x+b,'r--')
plt.xlabel('Actual position')
plt.ylabel('Decoded position')
plt.title(f'Decoder\nR2={decoder_stats[0].round(4)}, p={decoder_stats[1].round(4)}')

plt.subplot(224)
plt.hist(total_pred_losses)
plt.xlabel('Loss')
plt.ylabel('Number')
plt.title(f'Decoder\nmean loss:{avg_pred_loss.round(4)}')

plt.tight_layout()
# %%
