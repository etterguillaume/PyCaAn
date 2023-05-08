#%% Imports
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('plot_style.mplstyle')

from tqdm import tqdm
from functions.simulate import simulate_activity
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import chi2

#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Plot example activities
rec_length=20
num_bins=2
plt.figure(figsize=(1.5,1))
activity, variable = simulate_activity(recording_length=rec_length,
                                       num_bins=num_bins,
                                       ground_truth_info=1,
                                       sampling=1/num_bins)
plt.subplot(211)
plt.bar(np.arange(rec_length),variable,label='State', color='C6')
plt.ylabel('State')

plt.subplot(212)
plt.bar(np.arange(rec_length),~activity,label='Activity', color='C0')
plt.ylabel('Activity')
plt.tight_layout()
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_20length_2bins_info1_samplingEqual.pdf'))

#%% Plot example activities
rec_length=20
num_bins=2
plt.figure(figsize=(1.5,1))
activity, variable = simulate_activity(recording_length=rec_length,
                                       num_bins=num_bins,
                                       ground_truth_info=.1,
                                       sampling=1/num_bins)
plt.subplot(211)
plt.bar(np.arange(rec_length),variable,label='State', color='C6')
plt.ylabel('State')

plt.subplot(212)
plt.bar(np.arange(rec_length),~activity,label='Activity', color='C0')
plt.ylabel('Activity')
plt.tight_layout()
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_20length_2bins_info01_samplingEqual.pdf'))

#%% Effect of ground truth on metric
ground_truth_vec = np.linspace(0.001,1,100)
recording_length=10000
num_bins = 25
sampling_vec = 1/num_bins

MI_vals = np.zeros(len(ground_truth_vec))
AMI_vals = np.zeros(len(ground_truth_vec))
X2_vals = np.zeros(len(ground_truth_vec))

for i in tqdm(range(len(ground_truth_vec))):
    activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=num_bins,
                                       ground_truth_info=ground_truth_vec[i],
                                       sampling=sampling_vec)
    
    MI_vals[i] = mutual_info_score(activity,variable)
    AMI_vals[i] = adjusted_mutual_info_score(activity,variable, average_method='min')

plt.figure(figsize=(1,.75))
plt.plot(ground_truth_vec, MI_vals, label='MI', color='C0')
plt.plot(ground_truth_vec, AMI_vals, label='AMI', color='C6')
plt.xlabel('Ground truth info.')
plt.ylabel('Info.')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_length10000_25bins_infoVec_samplingEqual.pdf'))

#%% Effect of ground truth on metric
ground_truth_vec = np.linspace(0.001,1,100)
recording_length=10000
num_bins = 2
sampling_vec = 1/num_bins

MI_vals = np.zeros(len(ground_truth_vec))
AMI_vals = np.zeros(len(ground_truth_vec))
X2_vals = np.zeros(len(ground_truth_vec))

for i in tqdm(range(len(ground_truth_vec))):
    activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=num_bins,
                                       ground_truth_info=ground_truth_vec[i],
                                       sampling=sampling_vec)
    
    MI_vals[i] = mutual_info_score(activity,variable)
    AMI_vals[i] = adjusted_mutual_info_score(activity,variable, average_method='min')

plt.figure(figsize=(1,.75))
plt.plot(ground_truth_vec, MI_vals, label='MI', color='C0')
plt.plot(ground_truth_vec, AMI_vals, label='AMI', color='C6')
plt.xlabel('Ground truth info.')
plt.ylabel('Info.')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_length10000_2bins_infoVec_samplingEqual.pdf'))

#%% Effect of sampling on on metric
ground_truth_vec = 1
recording_length=10000
num_bins = 25
sampling_vec = np.linspace(0.001,1/num_bins,100)

MI_vals = np.zeros(len(sampling_vec))
AMI_vals = np.zeros(len(sampling_vec))
X2_vals = np.zeros(len(sampling_vec))

for i in tqdm(range(len(sampling_vec))):
    activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=num_bins,
                                       ground_truth_info=ground_truth_vec,
                                       sampling=sampling_vec[i])
    
    MI_vals[i] = mutual_info_score(activity,variable)
    AMI_vals[i] = adjusted_mutual_info_score(activity,variable, average_method='min')

plt.figure(figsize=(1,.75))
plt.plot(sampling_vec, MI_vals, label='MI', color='C0')
plt.plot(sampling_vec, AMI_vals, label='AMI', color='C6')
plt.xlabel('Portion samples')
plt.ylabel('Info.')
plt.xticks([0,1/num_bins],['0','1/$n_{bins}$'])
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_length10000_25bins_info1_samplingVec.pdf'))

#%% Effect of sampling on on metric: when there is no information
ground_truth_vec = 0
recording_length=10000
num_bins = 25
sampling_vec = np.linspace(0.001,1/num_bins,100)

MI_vals = np.zeros(len(sampling_vec))
AMI_vals = np.zeros(len(sampling_vec))
X2_vals = np.zeros(len(sampling_vec))

for i in tqdm(range(len(sampling_vec))):
    activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=num_bins,
                                       ground_truth_info=ground_truth_vec,
                                       sampling=sampling_vec[i])
    
    MI_vals[i] = mutual_info_score(activity,variable)
    AMI_vals[i] = adjusted_mutual_info_score(activity,variable, average_method='min')

plt.figure(figsize=(1,.75))
plt.plot(sampling_vec, MI_vals, label='MI', color='C0')
plt.plot(sampling_vec, AMI_vals, label='AMI', color='C6')
plt.xlabel('Portion samples')
plt.ylabel('Info.')
plt.xticks([0,1/num_bins],['0','1/$n_{bins}$'])
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_length10000_25bins_info0_samplingVec.pdf'))

#%% Effect of number of bins on on metric
ground_truth_vec = 0
recording_length=10000
num_bins = np.arange(2,200)

MI_vals = np.zeros(len(num_bins))
AMI_vals = np.zeros(len(num_bins))
X2_vals = np.zeros(len(num_bins))

for i in tqdm(range(len(num_bins))):
    sampling = 1/num_bins[i]
    activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=num_bins[i],
                                       ground_truth_info=ground_truth_vec,
                                       sampling=sampling)
    
    MI_vals[i] = mutual_info_score(activity,variable)
    AMI_vals[i] = adjusted_mutual_info_score(activity,variable, average_method='min')

plt.figure(figsize=(1,.75))
plt.title('Ground truth information: 0\nRecording length: 10,000 samples')
plt.plot(num_bins, MI_vals, label='MI', color='C0')
plt.plot(num_bins, AMI_vals, label='AMI', color='C6')
plt.xlabel('Number of bins')
plt.ylabel('Info.')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_length10000_binsVec_info0_samplingEqual.pdf'))

#%% Effect of number of bins on on metric with perfect information
# This can be used to estimate maximum information per n_bins
ground_truth_vec = 1
recording_length=10000
num_bins = np.arange(2,200)


MI_vals = np.zeros(len(num_bins))
AMI_vals = np.zeros(len(num_bins))
X2_vals = np.zeros(len(num_bins))

for i in tqdm(range(len(num_bins))):
    sampling = 1/num_bins[i]
    activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=num_bins[i],
                                       ground_truth_info=ground_truth_vec,
                                       sampling=sampling)
    
    MI_vals[i] = mutual_info_score(activity,variable)
    AMI_vals[i] = adjusted_mutual_info_score(activity,variable, average_method='min')

plt.figure(figsize=(1,.75))
plt.title('Ground truth information: 0\nRecording length: 10,000 samples')
plt.plot(num_bins, MI_vals, label='MI', color='C0')
plt.plot(num_bins, AMI_vals, label='AMI', color='C6')
plt.xlabel('Number of bins')
plt.ylabel('Info.')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_length10000_binsVec_info1_samplingEqual.pdf'))


#%% Behavioral variable with multiple bins
ground_truth_info = 1
sampling_vec = np.linspace(0.001,1,100)
recording_length=10000
bin_vec = np.arange(2,100)

MI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
AMI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
X2_mx = np.zeros((len(sampling_vec),len(bin_vec)))

for i in tqdm(range(len(bin_vec))):
    for j in range(len(sampling_vec)):
        activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=bin_vec[i],
                                       ground_truth_info=ground_truth_info,
                                       sampling=sampling_vec[j])
        
        MI_mx[j,i] = mutual_info_score(activity,variable)
        AMI_mx[j,i] = adjusted_mutual_info_score(activity,variable, average_method='min')
        X2_mx[j,i] = chi2(activity[:,None],variable[:,None])[1]

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(MI_mx,cmap='magma', interpolation='bicubic',origin='lower', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_binsVec_info1_samplingVec_MI.pdf'))

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(AMI_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='adjusted MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_binsVec_info1_samplingVec_AMI.pdf'))

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(X2_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=.05)
plt.colorbar(label='Chi2 p-value')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_binsVec_info1_samplingVec_Chi2.pdf'))


#%% Behavioral variable with multiple bins
ground_truth_info = 0
sampling_vec = np.linspace(0.001,1,100)
recording_length=10000
bin_vec = np.arange(2,100)

MI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
AMI_mx = np.zeros((len(sampling_vec),len(bin_vec)))
X2_mx = np.zeros((len(sampling_vec),len(bin_vec)))

for i in tqdm(range(len(bin_vec))):
    for j in range(len(sampling_vec)):
        activity, variable = simulate_activity(recording_length=recording_length,
                                       num_bins=bin_vec[i],
                                       ground_truth_info=ground_truth_info,
                                       sampling=sampling_vec[j])
        
        MI_mx[j,i] = mutual_info_score(activity,variable)
        AMI_mx[j,i] = adjusted_mutual_info_score(activity,variable, average_method='min')
        X2_mx[j,i] = chi2(activity[:,None],variable[:,None])[1]

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(MI_mx,cmap='magma', interpolation='bicubic',origin='lower', aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_binsVec_info0_samplingVec_MI.pdf'))

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(AMI_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto', vmin=0, vmax=1)
plt.colorbar(label='adjusted MI')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_binsVec_info0_samplingVec_AMI.pdf'))

plt.figure()
plt.title(f'Ground truth info: {ground_truth_info}')
plt.imshow(X2_mx,cmap='magma', interpolation='bicubic',origin='lower',aspect='auto',vmin=0,vmax=.05)
plt.colorbar(label='Chi2 p-value')
plt.xlabel('num. bins')
plt.ylabel('sampled portion')
plt.savefig(os.path.join(params['path_to_results'],'figures','sim_activity_binsVec_info0_samplingVec_Chi2.pdf'))
# %%