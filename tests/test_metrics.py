#%% Imports
from functions.analysis import reconstruction_accuracy
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Generate data
numNeurons = 100
recordingLength = 2000
threshold = .95
isNoiseAdditive = False

original = torch.rand((recordingLength,numNeurons))
original[original<=threshold] = 0
original[original>threshold] = 1
reconstruction = deepcopy(original)

#%% Parameterize noise injection
for neuron in range(numNeurons):
    noise_vec = torch.rand(recordingLength)
    if isNoiseAdditive:
        reconstruction[noise_vec>neuron/(numNeurons+1),neuron] = 1
    else:
        reconstruction[noise_vec>neuron/(numNeurons+1),neuron] = 0

#%%
accuracy, precision, recall, F1 = reconstruction_accuracy(reconstruction, original)

# %%
plt.figure(figsize=(4,4))
plt.subplot(3,2,1)
plt.imshow(original,aspect='auto', interpolation='none')
plt.title("original")

plt.subplot(3,2,2)
plt.imshow(reconstruction,aspect='auto', interpolation='none')
plt.title("reconstruction")

plt.subplot(3,2,3)
plt.plot(accuracy)
plt.title('accuracy')

plt.subplot(3,2,4)
plt.plot(F1)
plt.title('F1')

plt.subplot(3,2,5)
plt.plot(precision)
plt.title('precision')

plt.subplot(3,2,6)
plt.plot(recall)
plt.title('recall')

plt.tight_layout()
# %%
