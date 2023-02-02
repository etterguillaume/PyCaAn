#%% Import dependencies
import yaml
from tqdm import tqdm
import os
from functions.dataloaders import load_data
from functions.signal_processing import binarize_ca_traces,  interpolate_behavior, compute_velocity
import torch
from functions.datasets import generateDataset
from models.autoencoders import AE_MLP
from models.decoders import linear_decoder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
import umap.umap_ as umap
sigmoid = torch.nn.Sigmoid()

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

#%% Interpolate location
position = interpolate_behavior(data['position'], data['behavTime'], data['caTime'])

#%% Compute velocity
velocity, running_ts = compute_velocity(position, data['caTime'], params['speed_threshold'])

#%%
traces = data['caTrace']
ca_time = data['caTime']

rmv_immobility = False
if rmv_immobility:
    traces = traces[running_ts,:] # Transpose to get matrix = samples x neurons
    ca_time = ca_time[running_ts]
    velocity = velocity[running_ts]
    position = position[running_ts,0]
#%% Signal processing
if params['data_type'] == 'binarized':
    traces = binarize_ca_traces(traces,
                                          z_threshold=params['z_threshold'],
                                          sampling_frequency=params['sampling_frequency'])           

# %% Establish model and objective functions
model = AE_MLP(input_dim=data['numNeurons'],latent_dim=64,output_dim=2).to(device) # TODO: add to params
optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate']) # TODO: add to params
criterion = torch.nn.MSELoss()

#%% Normalize dataset and convert to tensor TODO normalize per neuron
dataset_mean=np.mean(traces,axis=1)
dataset_std=np.std(traces,axis=1)
norm_traces = (traces-dataset_mean)/dataset_std
norm_traces = torch.tensor(norm_traces,dtype=torch.float32).to(device)

#%%
dataset = generateDataset(norm_traces, position) #TODO add other variables

#%% Establish dataset TODO: add labels! eg. position, direction, velocity, time
train_test_ratio=.8 # Train/test ratio
dataset_size=len(dataset)
train_set_size = int(dataset_size * train_test_ratio)
test_set_size = dataset_size - train_set_size
train_set, test_set = random_split(dataset, [train_set_size, test_set_size]) # Random split

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
n_train = len(train_loader)
n_test = len(test_loader)

#%%
train_loss=[]
test_loss=[]
for epoch in tqdm(range(params["maxTrainSteps"])):
    run_train_loss = 0
    model.train()
    for i, (x, t) in enumerate(train_loader):
        x = x.to(device)
        reconstruction, _ = model(x)
        loss = criterion(reconstruction, x)
        loss.backward()
        optimizer.step()
        run_train_loss += loss.item()
        optimizer.zero_grad()
    train_loss.append(run_train_loss/n_train)

    run_test_loss = 0
    model.eval()
    for i, (x, t) in enumerate(test_loader):
        x = x.to(device)
        with torch.no_grad():
            reconstruction, _ = model(x)
            loss = criterion(reconstruction, x) # Only compute loss on masked part
            run_test_loss += loss.item()
    test_loss.append(run_test_loss/n_test)

    print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Test Loss: {run_test_loss/n_test:.4f}")    
    torch.save({
            'params': params,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': test_loss,
            }, f'results/AE_model.pt')
# %%
# Plot training curves
plt.plot(train_loss)
plt.plot(test_loss)

#%%
# Plot reconstruction examples
with torch.no_grad():
    reconstruction, embedding = model(norm_traces)

#%%
plt.subplot(121)
max_val=torch.max(norm_traces)
cells2plot = 10
for i in range(cells2plot):
    plt.plot(traces[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
            c=(1-i/50,.6,i/50),
            linewidth=.3)    
    plt.xlim([0,2000])
plt.title(f'Original: {params["AE_dropout_rate"]}')

max_val=torch.max(reconstruction)
plt.subplot(122)
for i in range(cells2plot):
    plt.plot(reconstruction[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
            c=(1-i/50,.6,i/50),
            linewidth=.3)
    plt.xlim([0,2000])
plt.title(f'Reconstruction\nDropout rate: {params["AE_dropout_rate"]}')
#plt.plot(datapoints[:,0]-reconstruction[:,0])

#%% UMAP for comparison
embedding_UMAP = umap.UMAP(n_neighbors=50, min_dist=0.8,n_components=2,metric='cosine').fit_transform(norm_traces)

#%%
interactive_plot_manifold3D(x=embedding_UMAP[:,0],z=embedding_UMAP[:,1],y=embedding_UMAP[:,1],color=embedding_UMAP[:,0])

#%% Train decoder on embedding and location
embedding_decoder = linear_decoder(input_dims=2)
UMAP_decoder = linear_decoder(input_dims=2)
# %%
decoder_optimizer = torch.optim.AdamW(embedding_decoder.parameters(), lr=params['learning_rate'])
UMAP_optimizer = torch.optim.AdamW(UMAP_decoder.parameters(), lr=params['learning_rate'])

#%%
train_loss=[]
test_loss=[]
for epoch in tqdm(range(params["maxTrainSteps"])):
    run_train_loss = 0
    model.eval()
    embedding_decoder.train()
    UMAP_decoder.train()
    for i, x in enumerate(train_loader):
        x = x.to(device)
        _, embedding = model(drop(x))
        pred = embedding_decoder(embedding)
        UMAP_pred = UMAP_decoder(embedding)
        decoder_loss = criterion(pred, position) # Only compute loss on masked part
        UMAP_decoder_loss = criterion(UMAP_pred, position)
        decoder_loss.backward()
        decoder_optimizer.step()
        run_train_loss += loss.item()
        optimizer.zero_grad()
    train_loss.append(run_train_loss/n_train)

    run_test_loss = 0
    model.eval()
    for i, x in enumerate(test_loader):
        x = x.to(device)
        perm = torch.randperm(x.size(1))
        idx = perm[:numMasked]
        with torch.no_grad():
            reconstruction, _ = model(drop(x))
            #if params['data_type']=='binarized':
            #    reconstruction=sigmoid(reconstruction)
            loss = criterion(reconstruction, x) # Only compute loss on masked part
            run_test_loss += loss.item()
    test_loss.append(run_test_loss/n_test)

    print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Test Loss: {run_test_loss/n_test:.4f}")    
    torch.save({
            'params': params,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': test_loss,
            }, f'results/AE_model.pt')


 # %%
