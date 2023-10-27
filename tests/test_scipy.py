#%% Tests for scipy functions
import numpy as np
from scipy.interpolate import griddata

#%% Generate data
X = np.array([[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]],dtype='float').T
val = np.arange(5)*10
print(X)
print(val)
# %%
griddata(X, val, xi)
# %%
