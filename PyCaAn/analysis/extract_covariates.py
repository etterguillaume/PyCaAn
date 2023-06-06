#%% Imports
import yaml
import os
from tqdm import tqdm
import numpy as np
from PyCaAn.functions.dataloaders import load_data
from PyCaAn.functions.signal_processing import preprocess_data
from PyCaAn.functions.tuning import assess_covariate
from PyCaAn.functions.metrics import extract_total_distance_travelled
import h5py

#%% Load parameters
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
    session_file = yaml.full_load(file)
session_list = session_file['sessions']
print(f'{len(session_list)} sessions to process')

# If tuning_data folder does not exist, create it
if not os.path.exists(params['path_to_results']):
    os.mkdir(params['path_to_results'])
if not os.path.exists(os.path.join(params['path_to_results'],'covariates')):
    os.mkdir(os.path.join(params['path_to_results'],'covariates'))

#%%
for i, session in enumerate(tqdm(session_list)):
    data = load_data(session)

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
    info_dict = {
                'day': data['day'],
                'task': data['task'],
                'subject': data['subject'],
                'region': data['region'],
                'sex': data['sex'],
                'age': data['age'],
                'condition': data['day'],
                'darkness': data['darkness'],
                'optoStim': data['optoStim'],
                'rewards': data['rewards'],
                'darkness': data['darkness'],
                'condition': data['condition'],
                'numNeurons': numNeurons,
                'numFrames': numFrames,
                'total_distance_travelled': float(total_distance_travelled),
                'duration': float(data['caTime'][-1]),
                'speed_threshold': params['speed_threshold']
        }
    if not os.path.exists(os.path.join(working_directory,'info.yaml')) or params['overwrite_mode']=='always':
        with open(os.path.join(working_directory,'info.yaml'),"w") as file:
            yaml.dump(info_dict,file)

    if not os.path.exists(os.path.join(working_directory,'covariates.h5')) or params['overwrite_mode']=='always':
        # Pre-allocate data for covariates
        info_matrix = np.ones((5,5))
        pvalue_matrix = np.zeros((5,5))
        labels=['space','time','distance','speed','heading']

        position=data['position']
        if data['task'] == 'OF':
            mazeSize=45
        elif data['task'] == 'legoOF':
            mazeSize=50
        elif data['task'] == 'plexiOF':
            mazeSize=49

        if data['task'] == 'LT':
            mazeSize=100
            position=data['position'][:,0]

        elif data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
            mazeSize=134
            position=data['position'][:,0]

        # Location vs time
        info_matrix[0,1], pvalue_matrix[0,1] = assess_covariate(
                position,
                data['elapsed_time'],
                data['running_ts'],
                mazeSize,
                params['spatialBinSize'],
                params['max_temporal_length'],
                params['temporalBinSize']
                )
        info_matrix[1,0], pvalue_matrix[1,0] = info_matrix[0,1], pvalue_matrix[0,1]

        # Location vs distance
        info_matrix[0,2], pvalue_matrix[0,2] = assess_covariate(
                position,
                data['distance_travelled'],
                data['running_ts'],
                mazeSize,
                params['spatialBinSize'],
                params['max_distance_length'],
                params['distanceBinSize']
                )
        info_matrix[2,0], pvalue_matrix[2,0] = info_matrix[0,2], pvalue_matrix[0,2]

        # Location vs velocity
        info_matrix[0,3], pvalue_matrix[0,3] = assess_covariate(
                position,
                data['velocity'],
                data['running_ts'],
                mazeSize,
                params['spatialBinSize'],
                params['max_velocity_length'],
                params['velocityBinSize']
                )
        info_matrix[3,0], pvalue_matrix[3,0] = info_matrix[0,3], pvalue_matrix[0,3]

        # Location vs velocity
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[0,4], pvalue_matrix[0,4] = assess_covariate(
                    position,
                    data['LT_direction'],
                    data['running_ts'],
                    mazeSize,
                    params['spatialBinSize'],
                    2,
                    1
                    )
            info_matrix[4,0], pvalue_matrix[4,0] = info_matrix[0,4], pvalue_matrix[0,4]
        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[0,4], pvalue_matrix[0,4] = assess_covariate(
                    position,
                    data['heading'],
                    data['running_ts'],
                    mazeSize,
                    params['spatialBinSize'],
                    360,
                    params['directionBinSize']
                    )
            info_matrix[4,0], pvalue_matrix[4,0] = info_matrix[0,4], pvalue_matrix[0,4]

        # Time vs distance
        info_matrix[1,2], pvalue_matrix[1,2] = assess_covariate(data['elapsed_time'],
                            data['distance_travelled'],
                            data['running_ts'],
                            params['max_temporal_length'],
                            params['temporalBinSize'],
                            params['max_distance_length'],
                            params['distanceBinSize'])
        info_matrix[2,1], pvalue_matrix[2,1] = info_matrix[1,2], pvalue_matrix[1,2]

        # Time vs speed
        info_matrix[1,3], pvalue_matrix[1,3] = assess_covariate(data['elapsed_time'],
                            data['velocity'],
                            data['running_ts'],
                            params['max_temporal_length'],
                            params['temporalBinSize'],
                            params['max_velocity_length'],
                            params['velocityBinSize'])
        info_matrix[3,1], pvalue_matrix[3,1] = info_matrix[1,3], pvalue_matrix[1,3]

        # Time vs heading
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[1,4], pvalue_matrix[1,4] = assess_covariate(data['elapsed_time'],
                                data['LT_direction'],
                                data['running_ts'],
                                params['max_temporal_length'],
                                params['temporalBinSize'],
                                2,
                                1
                                )
            info_matrix[4,1], pvalue_matrix[4,1] = info_matrix[1,4], pvalue_matrix[1,4]
        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[1,4], pvalue_matrix[1,4] = assess_covariate(data['elapsed_time'],
                                data['heading'],
                                data['running_ts'],
                                params['max_temporal_length'],
                                params['temporalBinSize'],
                                360,
                                params['directionBinSize']
                                )
            info_matrix[4,1], pvalue_matrix[4,1] = info_matrix[1,4], pvalue_matrix[1,4]

        # Distance vs speed
        info_matrix[2,3], pvalue_matrix[2,3] = assess_covariate(
                            data['distance_travelled'],
                            data['velocity'],
                            data['running_ts'],
                            params['max_distance_length'],
                            params['distanceBinSize'],
                            params['max_velocity_length'],
                            params['velocityBinSize'])
        info_matrix[3,2], pvalue_matrix[3,2] = info_matrix[2,3], pvalue_matrix[2,3]

        # distance vs heading
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[2,4], pvalue_matrix[2,4] = assess_covariate(
                                data['distance_travelled'],
                                data['LT_direction'],
                                data['running_ts'],
                                params['max_distance_length'],
                                params['distanceBinSize'],
                                2,
                                1
                                )
            info_matrix[4,2], pvalue_matrix[4,2] = info_matrix[2,4], pvalue_matrix[2,4]
        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[2,4], pvalue_matrix[2,4] = assess_covariate(
                                data['distance_travelled'],
                                data['heading'],
                                data['running_ts'],
                                params['max_distance_length'],
                                params['distanceBinSize'],
                                360,
                                params['directionBinSize']
                                )
            info_matrix[4,2], pvalue_matrix[4,2] = info_matrix[2,4], pvalue_matrix[2,4]

        # velocity vs heading
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[3,4], pvalue_matrix[3,4] = assess_covariate(
                                data['velocity'],
                                data['LT_direction'],
                                data['running_ts'],
                                params['max_velocity_length'],
                                params['velocityBinSize'],
                                2,
                                1
                                )
            info_matrix[4,3], pvalue_matrix[4,3] = info_matrix[3,4], pvalue_matrix[3,4]
        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[3,4], pvalue_matrix[3,4] = assess_covariate(
                                data['velocity'],
                                data['heading'],
                                data['running_ts'],
                                params['max_velocity_length'],
                                params['velocityBinSize'],
                                360,
                                params['directionBinSize']
                                )
            info_matrix[4,3], pvalue_matrix[4,3] = info_matrix[3,4], pvalue_matrix[3,4]

        with h5py.File(os.path.join(working_directory,'covariates.h5'),'w') as f:
            f.create_dataset('AMI', data=info_matrix)
            f.create_dataset('p_value', data=pvalue_matrix)
            f.create_dataset('labels', data=labels)
# %%
