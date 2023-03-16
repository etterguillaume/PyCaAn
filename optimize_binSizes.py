#%% Imports
import yaml
import os
import numpy as np
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data, extract_tone, extract_seqLT_tone
from functions.tuning import extract_1D_tuning, extract_2D_tuning, extract_discrete_tuning
from functions.metrics import extract_total_distance_travelled
import h5py
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

# Tested values (7 points)
time_bins = [.0625,.125,.25,.5,1,2,4]
space_bins = [.125,.25,.5,1,2,4,8]
speed_bins = [.125,.25,.5,1,2,4,8]
heading_bins = [0.5625,1.125, 2.25, 4.5, 9, 18, 36]

#%%
path = '../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621'

#%%
data = load_data(path)
data=preprocess_data(data,params) # Pre-process data
numFrames, numNeurons = data['rawData'].shape

#%% TIME
temporal_AMI = np.zeros((len(time_bins),numNeurons))

for bin_i in range(len(time_bins)): # hard-coded for 7 bins but oh well
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
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
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','spatial_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(temporal_AMI,axis=1)
error_band=np.std(temporal_AMI,axis=1)
plt.plot(time_bins, mean_line)
plt.xscale('log')
#plt.xticks(time_bins)
plt.xlabel('Time bins (s)')
plt.fill_between(time_bins, mean_line-error_band, mean_line+error_band,alpha=.2)
plt.plot([.25,.25],[0,.002])
plt.title('Ideal bin size = 0.25 s')

#%% SPACE
spatial_AMI = np.zeros((len(space_bins),numNeurons))

for bin_i in tqdm(range(len(space_bins))): # hard-coded for 7 bins but oh well
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_2D_tuning(data['binaryData'],
                                                            data['position'],
                                                            data['running_ts'],
                                                            var_length=45,
                                                            bin_size=space_bins[bin_i])
    spatial_AMI[bin_i,:]=AMI

#%% Plot
plt.figure(figsize=(2,1))
plt.imshow(spatial_AMI.T,aspect='auto', interpolation='none',cmap='magma')
plt.title(f'Max duration: {params["max_temporal_length"]}')
plt.xticks(np.arange(len(time_bins))[::2],time_bins[::2])
plt.xlabel('Time bins (s)')
plt.ylabel('Neuron ID')
plt.colorbar(label='AMI')

if not os.path.exists(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes')):
    os.mkdir(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes'))
plt.savefig(os.path.join(params['path_to_results'],'figures','optimal_bin_sizes','spatial_AMI.pdf'))
#%%
#plt.figure(figsize=(2,1))
mean_line=np.mean(spatial_AMI,axis=1)
error_band=np.std(spatial_AMI,axis=1)
plt.plot(time_bins, mean_line)
plt.xscale('log')
#plt.xticks(time_bins)
plt.xlabel('Time bins (s)')
plt.fill_between(time_bins, mean_line-error_band, mean_line+error_band,alpha=.2)
plt.plot([.25,.25],[0,.002])
plt.title('Ideal bin size = 0.25 s')













# Extract tuning to distance
if not os.path.exists(os.path.join(working_directory,'distance_tuning.h5')) or params['overwrite_mode']=='always':
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['distance_travelled'],
                                                        data['running_ts'],
                                                        var_length=params['max_distance_length'],
                                                        bin_size=params['spatialBinSize'])

    with h5py.File(os.path.join(working_directory,'distance_tuning.h5'),'w') as f:
        f.create_dataset('AMI', data=AMI)
        f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
        f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
        f.create_dataset('tuning_curves', data=tuning_curves)

# Extract tuning to velocity
if not os.path.exists(os.path.join(working_directory,'velocity_tuning.h5')) or params['overwrite_mode']=='always':
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['velocity'],
                                                        data['running_ts'],
                                                        var_length=params['max_velocity_length'],
                                                        bin_size=params['velocityBinSize'])

    with h5py.File(os.path.join(working_directory,'velocity_tuning.h5'),'w') as f:
        f.create_dataset('AMI', data=AMI)
        f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
        f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
        f.create_dataset('tuning_curves', data=tuning_curves)

# Extract spatial tuning
if not os.path.exists(os.path.join(working_directory,'spatial_tuning.h5')) or params['overwrite_mode']=='always':
    if data['task']=='OF':
        AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_2D_tuning(data['binaryData'],
                                                        data['position'],
                                                        data['running_ts'],
                                                        var_length=45,
                                                        bin_size=params['spatialBinSize'])
        
    elif data['task']=='legoOF':
        AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_2D_tuning(data['binaryData'],
                                                        data['position'],
                                                        data['running_ts'],
                                                        var_length=50,
                                                        bin_size=params['spatialBinSize'])

    elif data['task']=='LT':
        AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['position'][:,0],
                                                        data['running_ts'],
                                                        var_length=100,
                                                        bin_size=params['spatialBinSize'])
        
    elif data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
        AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['position'][:,0],
                                                        data['running_ts'],
                                                        var_length=134,
                                                        bin_size=params['spatialBinSize'])
        
    with h5py.File(os.path.join(working_directory,'spatial_tuning.h5'),'w') as f:
        f.create_dataset('AMI', data=AMI)
        f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
        f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
        f.create_dataset('tuning_curves', data=tuning_curves)

try:
    if not os.path.exists(os.path.join(working_directory,'direction_tuning.h5')) or params['overwrite_mode']=='always':
        if data['task'] == 'OF' or data['task'] == 'legoOF':
            AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['heading'],
                                                            data['running_ts'],
                                                            var_length=2, #TODO
                                                            bin_size=1) #TODO
            
        with h5py.File(os.path.join(working_directory,'direction_tuning.h5'),'w') as f:
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
except:
    print('Could not extract tuning to direction')
