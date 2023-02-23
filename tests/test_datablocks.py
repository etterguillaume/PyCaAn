#%%
import torch

#%%
a = torch.rand((102,50))
a.shape
# %%
b = torch.split(a, 10)
# %%
print([i for i in b])# %%

# %%
