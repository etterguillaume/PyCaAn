#%% Imports
import yaml
import os
from tqdm import tqdm
import numpy as np
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import assess_covariate
from functions.metrics import extract_total_distance_travelled
import h5py

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
    session_file = yaml.full_load(file)
session_list = session_file['sessions']
print(f'{len(session_list)} sessions to process')

#%%
for i, session in enumerate(tqdm(session_list)):
    data = load_data(session)

    # If tuning_data folder does not exist, create it
    if not os.path.exists(params['path_to_results']):
       os.mkdir(params['path_to_results'])
    if not os.path.exists(os.path.join(params['path_to_results'],'covariates')):
       os.mkdir(os.path.join(params['path_to_results'],'covariates'))

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        'covariates',
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    # Pre-process data
    data=preprocess_data(data,params)

    # Save basic info
    numFrames, numNeurons = data['rawData'].shape
    total_distance_travelled = extract_total_distance_travelled(data['position'])


if not os.path.exists(os.path.join(working_directory,'covariates.h5')) or params['overwrite_mode']=='always':
    # Pre-allocate data for covariates
    info_matrix = np.ones((5,5))
    pvalue_matrix = np.zeros((5,5))
    labels=['space','time','distance','speed','heading']

    # Time vs distance
    info_matrix[1,2], pvalue_matrix[1,2] = assess_covariate(data['elapsed_time'],
                        data['distance_travelled'],
                        data['running_ts'],
                        params['max_temporal_length'],
                        params['temporalBinSize'],
                        params['max_distance_length'],
                        params['distanceBinSize'])
    info_matrix[1,2], pvalue_matrix[1,2] = info_matrix[2,1], pvalue_matrix[2,1]

    # Time vs speed
    info_matrix[1,3], pvalue_matrix[1,3] = assess_covariate(data['elapsed_time'],
                        data['velocity'],
                        data['running_ts'],
                        params['max_temporal_length'],
                        params['temporalBinSize'],
                        params['max_velocity_length'],
                        params['velocityBinSize'])
    info_matrix[1,3], pvalue_matrix[1,3] = info_matrix[3,1], pvalue_matrix[3,1]

    # Distance vs speed
    info_matrix[2,3], pvalue_matrix[2,3] = assess_covariate(data['distance_travelled'],
                        data['velocity'],
                        data['running_ts'],
                        params['max_distance_length'],
                        params['temporalBinSize'],
                        params['max_velocity_length'],
                        params['distanceBinSize'])
    info_matrix[2,3], pvalue_matrix[2,3] = info_matrix[3,2], pvalue_matrix[3,2]

    if data['task'] == 'OF':
        info_matrix[0,1], pvalue_matrix[0,1] = assess_covariate(data['elapsed_time'],
                data['distance_travelled'],
                data['running_ts'],
                params['max_temporal_length'],
                params['temporalBinSize'],
                params['max_distance_length'],
                params['distanceBinSize'])

    with h5py.File(os.path.join(working_directory,'covariates.h5'),'w') as f:
        f.create_dataset('AMI', data=info_matrix)
        f.create_dataset('p_value', data=pvalue_matrix)
        f.create_dataset('labels', data=labels)