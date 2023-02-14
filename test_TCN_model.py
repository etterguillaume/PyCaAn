#%%
import torch
from models.autoencoders import TCN_10, TCN
import matplotlib.pyplot as plt

#%%
data = torch.randn((64,100,10)) # Create 10 frames x 100 neurons random datablock

#%%
test_conv1d = torch.nn.Conv1d(in_channels=100, out_channels=2, kernel_size=10, stride=1,padding=0)
test_conv2d = torch.nn.Conv2d(in_channels=100, out_channels=2, kernel_size=(10,1), stride=1,padding=0)

#%%
model = TCN_10(input_dim=100,output_dim=2)

# %%
reconstruction, embedding = model(data)

#%% Assert model output
assert reconstruction.shape[1] == data.shape[1], f"Expected model output channels to be {data.shape[1]} but got {reconstruction.shape[1]} instead"
assert reconstruction.shape[2] == data.shape[2], f"Expected model output channels to be {data.shape[2]} but got {reconstruction.shape[2]} instead"

#%%
plt.subplot(211)
plt.imshow(data[0,:,:],aspect='auto',interpolation='none'); plt.colorbar()
plt.subplot(212)
plt.imshow(reconstruction[0,:,:].detach(),aspect='auto',interpolation='none'); plt.colorbar()

# %%
from models.decoders import TCN10_decoder
# %%
decoder = torch.nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=10)

# %%
model = TCN(input_dim=100, hidden_dims = [64,32,16,8], output_dim=2, kernel_size=10)
model

# %%
reconstruction, embedding = model(data)

#%%
assert reconstruction.shape[1] == data.shape[1], f"Expected model output channels to be {data.shape[1]} but got {reconstruction.shape[1]} instead"
assert reconstruction.shape[2] == data.shape[2], f"Expected model output channels to be {data.shape[2]} but got {reconstruction.shape[2]} instead"

#%%
plt.subplot(211)
plt.imshow(data[0,:,:],aspect='auto',interpolation='none'); plt.colorbar()
plt.subplot(212)
plt.imshow(reconstruction[0,:,:].detach(),aspect='auto',interpolation='none'); plt.colorbar()
# %%
reconstruction.shape

# %%
