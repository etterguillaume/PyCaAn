#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Import dependencies
import yaml
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
import torch
from umap.umap_ import UMAP

import numpy as np

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '../../datasets/calcium_imaging/M246/M246_LT_6'
data = load_data(session_path)

#%% Preprocessing 
params['data_type'] = 'raw'
data = preprocess_data(data,params)
#%%
# Create chunks
numChunks = len(data['procData'])-params['data_block_size']+1 # Compute number of chunks
neural_data = np.zeros((numChunks, data['procData'].shape[1],params['data_block_size']))
position = np.zeros((numChunks,2,params['data_block_size']))
velocity = np.zeros((numChunks,1,params['data_block_size']))
for chunk in range(numChunks):
    neural_data[chunk,:,:] = data['procData'][chunk:chunk+params['data_block_size'],:].T
    position[chunk,:,:] = data['position'][chunk:chunk+params['data_block_size'],:].T
    velocity[chunk,0,:] = data['velocity'][chunk:chunk+params['data_block_size']]

#%%
embedding_raw_chunks = UMAP(n_neighbors=50, min_dist=0.0125,n_components=2,metric='cosine').fit_transform(neural_data.reshape(-1,data['procData'].shape[1]))

#%%
data = load_data(session_path)
params['data_type'] = 'binarized'
data = preprocess_data(data,params)

# Create chunks
numChunks = len(data['procData'])-params['data_block_size']+1 # Compute number of chunks
neural_data = np.zeros((numChunks, data['procData'].shape[1],params['data_block_size']))
position = np.zeros((numChunks,2,params['data_block_size']))
velocity = np.zeros((numChunks,1,params['data_block_size']))
for chunk in range(numChunks):
    neural_data[chunk,:,:] = data['procData'][chunk:chunk+params['data_block_size'],:].T
    position[chunk,:,:] = data['position'][chunk:chunk+params['data_block_size'],:].T
    velocity[chunk,0,:] = data['velocity'][chunk:chunk+params['data_block_size']]

#%%
embedding_binarized_chunks = UMAP(n_neighbors=50, min_dist=0.0125,n_components=2,metric='cosine').fit_transform(neural_data.reshape(-1,data['procData'].shape[1]))

#%%
data = load_data(session_path)

#%%
params['data_type'] = 'raw'
data = preprocess_data(data,params)

#%%
embedding_raw = UMAP(n_neighbors=50, min_dist=0.0125,n_components=2,metric='cosine').fit_transform(data['procData'])

#%%
data = load_data(session_path)
params['data_type'] = 'binarized'
data = preprocess_data(data,params)

#%%
embedding_binarized = UMAP(n_neighbors=50, min_dist=0.0125,n_components=2,metric='cosine').fit_transform(data['procData'])

# %% Position
plt.figure(figsize=(3,3))
plt.subplot(221)
plt.scatter(embedding_raw_chunks[:,0], embedding_raw_chunks[:,1],c=position.reshape(-1,2)[:,0])
plt.title('10F chunks\nraw')

plt.subplot(222)
plt.scatter(embedding_binarized_chunks[:,0], embedding_binarized_chunks[:,1],c=position.reshape(-1,2)[:,0])
plt.title('10F chunks\nbinarized')

plt.subplot(223)
plt.scatter(embedding_raw[:,0], embedding_raw[:,1],c=data['position'][:,0])
plt.title('1F raw')

plt.subplot(224)
plt.scatter(embedding_binarized[:,0], embedding_binarized[:,1],c=data['position'][:,0])
plt.title('1F binarized')
plt.tight_layout
# %% Velocity
plt.figure(figsize=(3,3))
plt.subplot(221)
plt.scatter(embedding_raw_chunks[:,0], embedding_raw_chunks[:,1],c=velocity)
plt.title('10F chunks\nraw')

plt.subplot(222)
plt.scatter(embedding_binarized_chunks[:,0], embedding_binarized_chunks[:,1],c=velocity)
plt.title('10F chunks\nbinarized')

plt.subplot(223)
plt.scatter(embedding_raw[:,0], embedding_raw[:,1],c=data['velocity'])
plt.title('1F raw')

plt.subplot(224)
plt.scatter(embedding_binarized[:,0], embedding_binarized[:,1],c=data['velocity'])
plt.title('1F binarized')
plt.tight_layout
# %%
