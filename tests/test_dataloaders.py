#%%
from .. import functions.dataloaders.load_data

#%%
path = '../../datasets/calcium_imaging/M986/M986_legoSeqLT_20190312'

data=load_data(path)
# %%
assert data is not None
# %%
