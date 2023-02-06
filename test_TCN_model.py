#%%
import torch
from models.autoencoders import TCN_10

#%%
model = TCN_10(input_dim=100,latent_dim=16,output_dim=2)
# %%
model
# %%
data = torch.randn((64,100,10)) # Create 10 frames x 100 neurons random datablock
# %%
embedding = model.encoder(data)
#%% Output
reconstruction = model.decoder(embedding)

#%% Assert model output
assert reconstruction.shape[1] == data.shape[1], f"Expected model output channels to be {data.shape[1]} but got {reconstruction.shape[1]} instead"
assert reconstruction.shape[2] == data.shape[2], f"Expected model output channels to be {data.shape[2]} but got {reconstruction.shape[2]} instead"

# %%
