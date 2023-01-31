#%% Import dependencies
import yaml
import tqdm
import os
from functions.dataloaders import load_data
import torch
from models.autoencoders import AE_MLP
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np

#TEMP
import matplotlib.pyplot as plt

#%%
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #TODO: define prefered device in params

device=torch.device('cpu')
seed = 42 # TODO add to params
torch.manual_seed(seed) # Seed for reproducibility
np.random.seed(seed)

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load session
session_path = '/Users/guillaumeetter/Documents/datasets/calcium_imaging/M246/M246_legoLT_20180716'
data = load_data(session_path)
# %%
data.keys()
# %%
data['caTrace'].shape

#%%
datapoints = torch.tensor(data['caTrace'].T,dtype=torch.float32).to(device)

#%%
plt.imshow(datapoints, interpolation=None, aspect='auto')

# %% Establish model and objective functions
model = AE_MLP(input_dim=data['numNeurons'],latent_dim=64,output_dim=3,DO_rate=0).to(device) # TODO: add to params
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # TODO: add to params
criterion = torch.nn.MSELoss()

#%%
reconstruction, embedding = model(datapoints)

#%%
plt.imshow(reconstruction.detach(), interpolation=None, aspect='auto')

#%% Establish dataset here
# Normalize
# Train/test sets
train_test_ratio=.75
generator=random_split()
n_train = int(np.round(train_test_ratio*datapoints.shape[0]))
split_idx = np.random.choice(np.arange(datapoints.shape[0]),n_train)
train_indices = np.zeros(datapoints.shape[0],dtype=bool)
train_indices[split_idx] = 1
train_loader = DataLoader(datapoints[train_indices,:], batch_size=64, shuffle=True)
test_loader = DataLoader(datapoints[test_indices,:], batch_size=64, shuffle=True)
n_train = len(train_loader)
n_val = len(test_loader)



#%%
early_stop = 0
for epoch in tqdm(range(50)):
    running_loss = 0
    for i, x in enumerate(train_loader):
        datapoints = [].to(device)
        # images = normalize(torch.cat([x_clean0, x_clean1])).to(device).float()
        reconstruction, _ = model(datapoints)
        loss = criterion(reconstruction, datapoints)
        running_loss += loss/x_clean0.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = 0
    for i, x in enumerate(test_loader):
        datapoints = [].to(device)
        # images = normalize(torch.cat([x_clean0, x_clean1])).to(device).float()
        with torch.no_grad():
            rec, x, mu, logvar = model(datapoints)                    
            loss = criterion(reconstruction, datapoints)
            val_loss += loss/x_clean0.size(0)

    if running_loss/n_train < val_loss/n_val:
        early_stop += 1
    print(f"Epoch: {epoch+1} \t Train Loss: {running_loss/n_train:.2f} \t Val Loss: {val_loss/n_val:.2f}")
    
    torch.save(model.state_dict(), params['path_to_results']+ '/models' + f'/{seed}/vae_{h_dim}_{latent_dim}.pth')
    if early_stop == patience:
        print("early stopping...")
        break