#%% Imports
import yaml
import os
import numpy as np
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import extract_1D_tuning, extract_2D_tuning
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Load parameters
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

# Tested values (7 points)
time_bins = [.0625,.125,.25,.5,1,2,4]
space_bins = [1,2,4,8,16]
distance_bins = [.125,.25,.5,1,2,4,8,16,32,64]
speed_bins = [.125,.25,.5,1,2,4,8]
heading_bins = [0.5625,1.125, 2.25, 4.5, 9, 18, 36]
#TODO find ideal bins using max (and/or diff b/w mean and median)
#TODO save results
#TODO save plots

#%%
path = '../../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621'

#%%
data = load_data(path)
data=preprocess_data(data,params) # Pre-process data
numFrames, numNeurons = data['rawData'].shape

#%% TIME
temporal_AMI = np.zeros((len(time_bins),numNeurons))

for bin_i in range(len(time_bins)): # hard-coded for 7 bins but oh well
    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_1D_tuning(data['binaryData'],
                                                            data['elapsed_time'],
                                                            data['running_ts'],
                                                            var_length=params['max_temporal_length'],
                                                            bin_size=time_bins[bin_i])
    temporal_AMI[bin_i,:]=AMI

#%% Plot
plt.figure(figsize=(2,1))
plt.imshow(temporal_AMI.T,aspect='auto', interpolation='none',cmap='magma')
plt.title(f'Max duration: {params["max_temporal_length"]}')
plt.xticks(np.arange(len(time_bins))[::2],time_bins[::2])
plt.xlabel('Time bins (s)')
plt.ylabel('Neuron ID')
plt.colorbar(label='AMI')

if not os.path.exists(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes')):
    os.mkdir(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes'))
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','temporal_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(temporal_AMI,axis=1)
median_line=np.median(temporal_AMI,axis=1)
error_band=np.std(temporal_AMI,axis=1)
plt.plot(time_bins, mean_line)
plt.fill_between(time_bins, mean_line-error_band, mean_line+error_band,alpha=.2)

plt.plot(time_bins, median_line)
plt.fill_between(time_bins, median_line-error_band, median_line+error_band,alpha=.2)

plt.xscale('log')
#plt.xticks(time_bins)
plt.ylabel('AMI')
plt.xlabel('Time bins (s)')

plt.plot([.125,.125],[0,.08],color='C6')
plt.title('Ideal bin size = 125 ms')
plt.legend()
plt.legend()

#%% SPACE
spatial_AMI = np.zeros((len(space_bins),numNeurons))

for bin_i in tqdm(range(len(space_bins))):
    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_2D_tuning(data['binaryData'],
                                                            data['position'],
                                                            data['running_ts'],
                                                            var_length=45,
                                                            bin_size=space_bins[bin_i])
    spatial_AMI[bin_i,:]=AMI

#%% Plot
plt.figure(figsize=(2,1))
plt.imshow(spatial_AMI.T,aspect='auto', interpolation='none',cmap='magma')
plt.title(f'Maze length: {45}')
plt.ylabel('AMI')
plt.xticks(np.arange(len(space_bins)),space_bins)
plt.xlabel('Space bins (cm)')
plt.ylabel('Neuron ID')
plt.colorbar(label='AMI')

if not os.path.exists(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes')):
    os.mkdir(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes'))
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','spatial_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(spatial_AMI,axis=1)
median_line=np.median(spatial_AMI,axis=1)
error_band=np.std(spatial_AMI,axis=1)
plt.plot(space_bins, mean_line)
plt.fill_between(space_bins, mean_line-error_band, mean_line+error_band,alpha=.2)

plt.plot(space_bins, median_line)
plt.fill_between(space_bins, median_line-error_band, median_line+error_band,alpha=.2)
plt.legend()
plt.xscale('log')
#plt.xticks(time_bins)
plt.xlabel('Space bins (cm)')
plt.ylabel('AMI')

plt.plot([4,4],[0,.25],color='C6')
plt.title('Ideal bin size = 4cm')
plt.legend()

#%% DISTANCE
distance_AMI = np.zeros((len(distance_bins),numNeurons))

for bin_i in range(len(distance_bins)): # hard-coded for 7 bins but oh well
    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['distance_travelled'],
                                                            data['running_ts'],
                                                            var_length=128,
                                                            bin_size=distance_bins[bin_i])
    distance_AMI[bin_i,:]=AMI

#%% Plot
plt.figure(figsize=(2,1))
plt.imshow(distance_AMI.T,aspect='auto', interpolation='none',cmap='magma')
plt.title(f'Max distance: {128}')
plt.xticks(np.arange(len(distance_bins))[::2],distance_bins[::2])
plt.xlabel('Distance bins (cm)')
plt.ylabel('Neuron ID')
plt.colorbar(label='AMI')

if not os.path.exists(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes')):
    os.mkdir(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes'))
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','distance_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(distance_AMI,axis=1)
median_line=np.median(distance_AMI,axis=1)
error_band=np.std(distance_AMI,axis=1)
plt.plot(distance_bins, mean_line)
plt.fill_between(distance_bins, mean_line-error_band, mean_line+error_band,alpha=.2)

plt.plot(distance_bins, median_line)
plt.fill_between(distance_bins, median_line-error_band, median_line+error_band,alpha=.2)

plt.xticks(distance_bins)
plt.xlabel('Distance bins (cm)')
plt.ylabel('AMI')

plt.xscale('log')
plt.plot([2,2],[0,.1],color='C6')
plt.title('Ideal bin size = 2 cm')
plt.legend()

#%% SPEED
speed_AMI = np.zeros((len(speed_bins),numNeurons))

for bin_i in range(len(speed_bins)): # hard-coded for 7 bins but oh well
    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['velocity'],
                                                            data['running_ts'],
                                                            var_length=30,
                                                            bin_size=speed_bins[bin_i])
    speed_AMI[bin_i,:]=AMI

#%% Plot
plt.figure(figsize=(2,1))
plt.imshow(speed_AMI.T,aspect='auto', interpolation='none',cmap='magma')
plt.title(f'Max speed: {30}')
plt.xticks(np.arange(len(speed_bins))[::2],speed_bins[::2])
plt.xlabel('Speed bins (cm.s$^{-1}$)')
plt.ylabel('Neuron ID')
plt.colorbar(label='AMI')

if not os.path.exists(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes')):
    os.mkdir(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes'))
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','speed_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(speed_AMI,axis=1)
median_line=np.median(speed_AMI,axis=1)
error_band=np.std(speed_AMI,axis=1)
plt.plot(speed_bins, mean_line)
plt.fill_between(speed_bins, mean_line-error_band, mean_line+error_band,alpha=.2)
plt.plot(speed_bins, median_line)
plt.fill_between(speed_bins, median_line-error_band, median_line+error_band,alpha=.2)

plt.xticks(speed_bins)
plt.xlabel('Speed bins (cm)')
plt.ylabel('AMI')
plt.xscale('log')
plt.plot([1,1],[0,.04], color='C6')
plt.title('Ideal bin size = 1 cm.s$^{-1}$')

#%% HEADING
heading_AMI = np.zeros((len(heading_bins),numNeurons))

for bin_i in range(len(heading_bins)): # hard-coded for 7 bins but oh well
    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['heading'],
                                                            data['running_ts'],
                                                            var_length=360,
                                                            bin_size=heading_bins[bin_i])
    heading_AMI[bin_i,:]=AMI

#%% Plot
plt.figure(figsize=(2,1))
plt.imshow(heading_AMI.T,aspect='auto', interpolation='none',cmap='magma')
plt.title(f'Max heading: {360}')
plt.xticks(np.arange(len(heading_bins))[::2],heading_bins[::2])
plt.xlabel('heading bins (º)')
plt.ylabel('Neuron ID')
plt.colorbar(label='AMI')

if not os.path.exists(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes')):
    os.mkdir(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes'))
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','heading_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(heading_AMI,axis=1)
median_line=np.median(heading_AMI,axis=1)
error_band=np.std(heading_AMI,axis=1)
plt.plot(heading_bins, mean_line,label="mean")
plt.fill_between(heading_bins, mean_line-error_band, mean_line+error_band,alpha=.2)
plt.plot(heading_bins, median_line,label="median")
plt.fill_between(heading_bins, median_line-error_band, median_line+error_band,alpha=.2)
plt.xticks(speed_bins)
plt.xlabel('heading bins (º)')
plt.ylabel('AMI')
plt.xscale('log')
plt.plot([4.5,4.5],[0,.1], color='C6')
plt.title('Ideal bin size = 4.5º')
# %%
