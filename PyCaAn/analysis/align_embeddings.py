#%% Imports
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load existing manifolds

# List possible combination (only do one side of the symmetry and no diagonal)
# n_possibilities = n_sessions**2/2-n_sessions
#%%
x=np.arange(1,250)
plt.plot(x, (x**2)/2-x) # plot to check how painful computations will be

# %%
manifold_a, manifold_b, var

for each value of (var)
    train linear transform between manifold_a and manfifold_b