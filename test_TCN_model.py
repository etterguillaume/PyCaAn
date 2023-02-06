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
# %%
test_TCN1 = torch.nn.ConvTranspose1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=0)

# %%
test_recon1=test_TCN1(embedding)
# %%
