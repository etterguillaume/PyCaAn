#%%
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import extract_tone, preprocess_data, clean_timestamps
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
#plt.style.use('plot_style.mplstyle')
#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%%
data = load_data('../' + params['path_to_dataset']+'/CA1/M986/M986_legoLT_20190201')
data = preprocess_data(data, params)

#%%
# Correlated pixels
plt.imshow(data['corrProj'].T, cmap='viridis', vmin=0, vmax=1)
plt.axis('off')
# plt.title('Correlated pixels')
cax = plt.axes([.95, 0.15, 0.05, 0.7])
plt.colorbar(cax=cax, label='Pixel correlation')
plt.savefig('/Users/guillaumeetter/Desktop/RSC_corrProj.pdf')

#%% Intro transients
import matplotlib
#cmap = matplotlib.cm.get_cmap('nipy_spectral')
cmap = matplotlib.cm.get_cmap('viridis')
#numNeurons=data['SFPs'].shape[0]
numNeurons=100
plt.figure(figsize=(2,2))
for i in range(numNeurons):
    color = cmap(i/(numNeurons))
    plt.plot(data['caTime'],data['neuralData'][:,i]/2+i,
            c=color,
            linewidth=.3, rasterized=False)

plt.xlim(0,60)
plt.xticks([0,30,60])
#plt.yticks([0,400],[0,200])
plt.ylim(0,numNeurons)
plt.xlabel('Time (s)')
plt.ylabel('Neuron #')

plt.tight_layout()
plt.savefig('/Users/guillaumeetter/Desktop/RSC_traces.pdf')
# %%
