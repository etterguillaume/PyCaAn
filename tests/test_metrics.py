#%% Imports
#from functions.analysis import reconstruction_accuracy
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import extract_tone, preprocess_data, clean_timestamps
import yaml
import os
from pycaan.functions.metrics import extract_firing_properties
from tqdm import tqdm
from sklearn.metrics import median_absolute_error as MAE

#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%%
data = load_data('../' + params['path_to_dataset']+'/CA1/M246/M246_OF_1')
#%%
data = preprocess_data(data, params)

#%% Assess MAE for angles
heading=data['heading']
pred_heading=np.mod(heading+90,360)
vanilla_MAE = MAE(heading,pred_heading)


#%%
marginal_likelihood, prob_off_to_on, prob_on_to_off = extract_firing_properties(CA3_data['binaryData'])

#%%
numFrames, numNeurons=CA3_data['binaryData'].shape
#%%
sum(np.diff(CA3_data['binaryData'][:,2].astype('int'))>0)/(numFrames-1)




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
