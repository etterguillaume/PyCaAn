#%% Imports
#from functions.analysis import reconstruction_accuracy
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('plot_style.mplstyle')

from tqdm import tqdm
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import chi2


#%% Binary variable
recording_length=1000
ground_truth_info = 1
sampling_vec = np.linspace(0.1,1,10)
activity_prob = np.linspace(0.1,1,10) # How likely will the neuron fire

MI_mx = np.zeros((len(sampling_vec),len(activity_prob)))
NMI_mx = np.zeros((len(sampling_vec),len(activity_prob)))
AMI_mx = np.zeros((len(sampling_vec),len(activity_prob)))
X2_mx = np.zeros((len(sampling_vec),len(activity_prob)))

for i in tqdm(range(len(activity_prob))):
    for j in range(len(sampling_vec)):
        activity = np.zeros(recording_length)
        behav_var = np.zeros(recording_length)
        activity_idx = np.random.choice(np.arange(recording_length),int(recording_length*activity_prob[i]),replace=False)
        activity[activity_idx]=True # Set active frames

        predict_behav_idx=np.random.choice(activity_idx,int(len(activity_idx)*sampling_vec[j]),replace=False)
        predict_behav_idx=np.random.choice(predict_behav_idx,int(len(predict_behav_idx)*ground_truth_info),replace=False)
        behav_var[predict_behav_idx] = True
        
        MI_mx[j,i] = mutual_info_score(activity,behav_var)
        NMI_mx[j,i] = normalized_mutual_info_score(activity,behav_var)
        AMI_mx[j,i] = adjusted_mutual_info_score(activity,behav_var)
        X2_mx[j,i] = chi2(activity[:,None],behav_var[:,None])[1]

#%% Plot
plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(MI_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=1)
plt.colorbar(label='MI')
plt.xlabel('activity probability')
plt.ylabel('sampled portion')
plt.xticks(np.arange(len(activity_prob))[::5],activity_prob[::5])
plt.yticks(np.arange(len(sampling_vec))[::5],sampling_vec[::5])

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(NMI_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=1)
plt.colorbar(label='normalized MI')
plt.xlabel('activity probability')
plt.ylabel('sampled portion')
plt.xticks(np.arange(len(activity_prob))[::5],activity_prob[::5])
plt.yticks(np.arange(len(sampling_vec))[::5],sampling_vec[::5])

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(AMI_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=1)
plt.colorbar(label='adjusted MI')
plt.xlabel('activity probability')
plt.ylabel('sampled portion')
plt.xticks(np.arange(len(activity_prob))[::5],activity_prob[::5])
plt.yticks(np.arange(len(sampling_vec))[::5],sampling_vec[::5])

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(X2_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=1)
plt.colorbar(label='Chi2 p-value')
plt.xlabel('activity probability')
plt.ylabel('sampled portion')
plt.xticks(np.arange(len(activity_prob))[::5],activity_prob[::5])
plt.yticks(np.arange(len(sampling_vec))[::5],sampling_vec[::5])
#%% Behavioral variable with multiple bins
ground_truth_info = 1
activity_prob = .01
sampling_vec = np.linspace(0.001,1,100)
recording_length=1000
bin_vec = np.arange(2,100)

MI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
NMI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
AMI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
X2_mx = np.zeros((len(sampling_vec),len(bin_vec)))

for i in tqdm(range(len(bin_vec))):
    for j in range(len(sampling_vec)):
        activity = np.zeros(recording_length,dtype='bool')
        behav_var = np.random.choice(np.arange(1,bin_vec[i]),recording_length) # randomly sample bins
        activity_idx = np.random.choice(np.arange(recording_length),int(recording_length*activity_prob),replace=False)
        activity[activity_idx]=True # Set active frames

        predict_behav_idx=np.random.choice(activity_idx,int(len(activity_idx)*sampling_vec[j]),replace=False)
        predict_behav_idx=np.random.choice(predict_behav_idx,int(len(predict_behav_idx)*ground_truth_info),replace=False)
        behav_var[predict_behav_idx] = 0 # bin zero predicts neural activity
        
        MI_mx[j,i] = mutual_info_score(activity,behav_var)
        NMI_mx[j,i] = normalized_mutual_info_score(activity,behav_var)
        AMI_mx[j,i] = adjusted_mutual_info_score(activity,behav_var)
        X2_mx[j,i] = chi2(activity[:,None],behav_var[:,None])[1]

#%%
plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(MI_mx,cmap='magma', norm=LogNorm(vmin=0.0001, vmax=.1), interpolation='bicubic',origin='lower', aspect='auto')
plt.colorbar(label='MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(NMI_mx,cmap='magma', norm=LogNorm(vmin=0.0001, vmax=.1), interpolation='bicubic',origin='lower',aspect='auto')
plt.colorbar(label='normalized MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(AMI_mx,cmap='magma', norm=LogNorm(vmin=0.0001, vmax=.1), interpolation='bicubic',origin='lower',aspect='auto')
plt.colorbar(label='adjusted MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(X2_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=.05)
plt.colorbar(label='Chi2 p-value')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')



#%%
activity_prob = .01
info_vec = np.linspace(0.001,1,100)
sampling_vec = np.linspace(0.001,1,100)
recording_length=10000
numBins = 10

MI_mx = np.zeros((len(sampling_vec),len(info_vec)))
NMI_mx = np.zeros((len(sampling_vec),len(info_vec)))
AMI_mx = np.zeros((len(sampling_vec),len(info_vec)))
X2_mx = np.zeros((len(sampling_vec),len(info_vec)))

for i in tqdm(range(len(info_vec))):
    for j in range(len(sampling_vec)):
        activity = np.zeros(recording_length,dtype='bool')
        behav_var = np.random.choice(np.arange(1,numBins),recording_length) # randomly sample bins
        activity_idx = np.random.choice(np.arange(recording_length),int(recording_length*activity_prob),replace=False)
        activity[activity_idx]=True # Set active frames

        predict_behav_idx=np.random.choice(activity_idx,int(len(activity_idx)*sampling_vec[j]),replace=False)
        predict_behav_idx=np.random.choice(predict_behav_idx,int(len(predict_behav_idx)*info_vec[i]),replace=False)
        behav_var[predict_behav_idx] = 0 # bin zero predicts neural activity
        
        MI_mx[j,i] = mutual_info_score(activity,behav_var)
        NMI_mx[j,i] = normalized_mutual_info_score(activity,behav_var)
        AMI_mx[j,i] = adjusted_mutual_info_score(activity,behav_var)
        X2_mx[j,i] = chi2(activity[:,None],behav_var[:,None])[1]

#%%

plt.figure()
plt.title('MI')
plt.imshow(MI_mx,cmap='magma', norm=LogNorm(vmin=0.0001, vmax=.1), interpolation='bicubic',origin='lower', aspect='auto')
plt.colorbar(label='MI')
plt.xlabel('true info.')
plt.ylabel('sampled portion')

plt.figure()
plt.title('NMI')
plt.imshow(NMI_mx,cmap='magma', norm=LogNorm(vmin=0.0001, vmax=.1), interpolation='bicubic',origin='lower',aspect='auto')
plt.colorbar(label='normalized MI')
plt.xlabel('true info.')
plt.ylabel('sampled portion')

plt.figure()
plt.title('AMI')
plt.imshow(AMI_mx,cmap='magma', norm=LogNorm(vmin=0.0001, vmax=.1), interpolation='bicubic',origin='lower',aspect='auto')
plt.colorbar(label='adjusted MI')
plt.xlabel('true info.')
plt.ylabel('sampled portion')

plt.figure()
plt.title('Chi2')
plt.imshow(X2_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto')
plt.colorbar(label='Chi2 p-value')
plt.xlabel('true info.')
plt.ylabel('sampled portion')










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
