#%% Import dependencies
import yaml
from tqdm import tqdm
import os
from functions.dataloaders import load_data
from functions.signal_processing import binarize_ca_traces,  interpolate_behavior
import torch
from models.autoencoders import AE_MLP
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
sigmoid = torch.nn.Sigmoid()

# Metrics
BCE_error = torch.nn.BCELoss(reduction='none')

#TEMP
import matplotlib.pyplot as plt

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Specify device and seeds
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #TODO: define prefered device in params
device = torch.device('cpu')

torch.manual_seed(params['seed']) # Seed for reproducibility
np.random.seed(params['seed'])

#%% Housekeeping
if not os.path.exists(params['path_to_results'] + '/models'):
    os.makedirs(params['path_to_results'] + '/models')

#%% Load session
session_path = '/Users/guillaumeetter/Documents/datasets/calcium_imaging/M246/M246_legoLT_20180716'
data = load_data(session_path)

#%% Interpolate location
position = interpolate_behavior(data['position'], data['behavTime'], data['caTime']).T

#%% Signal processing
traces = data['caTrace'].T # Transpose to get matrix = samples x neurons
if params['data_type'] == 'binarized':
    traces = binarize_ca_traces(traces,
                                          z_threshold=params['z_threshold'],
                                          sampling_frequency=params['sampling_frequency']).T            

#%%
# If binarized, no need to normalize. Else, apply transform here
traces = torch.tensor(traces,dtype=torch.float32).to(device)

# %% Establish model and objective functions
DO_rate=.5 #TODO: add to params
model = AE_MLP(input_dim=data['numNeurons'],latent_dim=64,output_dim=2,DO_rate=DO_rate).to(device) # TODO: add to params
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # TODO: add to params
if params['data_type'] == 'binarized':
    criterion = torch.nn.BCELoss()
else:
    criterion = torch.nn.MSELoss()

#%% Establish dataset
train_test_ratio=.75 # Train/test ratio
dataset_size=traces.shape[0]
train_set_size = int(dataset_size * train_test_ratio)
test_set_size = dataset_size - train_set_size
train_set, test_set = random_split(traces, [train_set_size, test_set_size]) # Random split

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
    for i, x in enumerate(train_loader):
        x = x.to(device)
        reconstruction, _ = model(x)
        if params['data_type']=='binarized':
            reconstruction=sigmoid(reconstruction)
        loss = criterion(reconstruction, x)
        loss.backward()
        optimizer.step()
        run_train_loss += loss.item()
        optimizer.zero_grad()
    train_loss.append(run_train_loss/n_train)

    run_test_loss = 0
    model.eval()
    for i, x in enumerate(test_loader):
        x = x.to(device)
        with torch.no_grad():
            reconstruction, _ = model(x)
            if params['data_type']=='binarized':
                reconstruction=sigmoid(reconstruction)
            loss = criterion(reconstruction, x)
            run_test_loss += loss.item()
    test_loss.append(run_test_loss/n_test)

    print(f"Epoch: {epoch+1} \t Train Loss: {run_train_loss/n_train:.4f} \t Val Loss: {run_test_loss/n_test:.4f}")    
    torch.save(model.state_dict(), params['path_to_results']+ '/models' + f'/model.pth')
# %%
# Plot training curves
plt.plot(train_loss)
plt.plot(test_loss)

#%%
# Plot reconstruction examples
with torch.no_grad():
    reconstruction, embedding = model(traces)

#%%
plt.subplot(121)
max_val=torch.max(traces)
cells2plot = 10
for i in range(cells2plot):
    plt.plot(traces[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
            c=(1-i/50,.6,i/50),
            linewidth=.3)    
    plt.xlim([0,2000])
plt.title(f'Original: {DO_rate}')

max_val=torch.max(reconstruction)
plt.subplot(122)
for i in range(cells2plot):
    plt.plot(reconstruction[:,i]*params['plot_gain']+max_val*i/params['plot_gain'],
            c=(1-i/50,.6,i/50),
            linewidth=.3)
    plt.xlim([0,2000])
plt.title(f'Reconstruction\nDropout rate: {DO_rate}')
#plt.plot(datapoints[:,0]-reconstruction[:,0])
# %%
embedding.shape
# %%
plt.scatter(embedding[:,0], embedding[:,1], c=position[:,0])
# %%
error = BCE_error(traces[:,0],reconstruction[:,0])
plt.plot(error)
# %%
