#%%
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

# %%
plt.figure
plt.plot(np.random.randn(100))
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.savefig('test.pdf')

# %%
