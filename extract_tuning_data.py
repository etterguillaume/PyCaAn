#%% Imports
import yaml
import os
from tqdm import tqdm
import numpy as np
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data, extract_tone, extract_seqLT_tone
from functions.tuning import extract_1D_tuning, extract_2D_tuning, extract_discrete_tuning
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
    if not os.path.exists(os.path.join(params['path_to_results'],'tuning_data')):
       os.mkdir(os.path.join(params['path_to_results'],'tuning_data'))

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        'tuning_data',
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

    # Extract tuning to time
    if not os.path.exists(os.path.join(working_directory,'temporal_tuning.h5')) or params['overwrite_mode']=='always':
        AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['elapsed_time'],
                                                            data['running_ts'],
                                                            var_length=params['max_temporal_length'],
                                                            bin_size=params['temporalBinSize'])

        with h5py.File(os.path.join(working_directory,'temporal_tuning.h5'),'w') as f:
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)

    # Extract tuning to distance
    if not os.path.exists(os.path.join(working_directory,'distance_tuning.h5')) or params['overwrite_mode']=='always':
        AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['distance_travelled'],
                                                            data['running_ts'],
                                                            var_length=params['max_distance_length'],
                                                            bin_size=params['spatialBinSize'])

        with h5py.File(os.path.join(working_directory,'distance_tuning.h5'),'w') as f:
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)

    # Extract tuning to velocity
    if not os.path.exists(os.path.join(working_directory,'velocity_tuning.h5')) or params['overwrite_mode']=='always':
        AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['velocity'],
                                                            data['running_ts'],
                                                            var_length=params['max_velocity_length'],
                                                            bin_size=params['velocityBinSize'])

        with h5py.File(os.path.join(working_directory,'velocity_tuning.h5'),'w') as f:
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)

    # Extract spatial tuning
    if not os.path.exists(os.path.join(working_directory,'spatial_tuning.h5')) or params['overwrite_mode']=='always':
        if data['task']=='OF':
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_2D_tuning(data['binaryData'],
                                                            data['position'],
                                                            data['running_ts'],
                                                            var_length=45,
                                                            bin_size=params['spatialBinSize'])
            
        elif data['task']=='legoOF':
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_2D_tuning(data['binaryData'],
                                                            data['position'],
                                                            data['running_ts'],
                                                            var_length=50,
                                                            bin_size=params['spatialBinSize'])

        elif data['task']=='LT':
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['position'][:,0],
                                                            data['running_ts'],
                                                            var_length=100,
                                                            bin_size=params['spatialBinSize'])
            
        elif data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                            data['position'][:,0],
                                                            data['running_ts'],
                                                            var_length=134,
                                                            bin_size=params['spatialBinSize'])
            
        with h5py.File(os.path.join(working_directory,'spatial_tuning.h5'),'w') as f:
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)

    # Extract direction tuning
    try:
        if not os.path.exists(os.path.join(working_directory,'direction_tuning.h5')) or params['overwrite_mode']=='always':
            if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF':
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                                data['heading'],
                                                                data['running_ts'],
                                                                var_length=360,
                                                                bin_size=params['directionBinSize'])
            elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                                data['LT_direction'],
                                                                data['running_ts'],
                                                                var_length=2,
                                                                bin_size=1)
                
            with h5py.File(os.path.join(working_directory,'direction_tuning.h5'),'w') as f:
                f.create_dataset('AMI', data=AMI)
                f.create_dataset('p_value', data=p_value)
                f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                f.create_dataset('tuning_curves', data=tuning_curves)
    except:
        print('Could not extract tuning to direction')


    # Extract tuning to tone
    if data['task'] == 'legoToneLT':
        try:
            if not os.path.exists(os.path.join(working_directory,'tone_tuning.h5')) or params['overwrite_mode']=='always':
                data=extract_tone(data,params)
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_discrete_tuning(data['binaryData'],
                                                                data['binaryTone'],
                                                                data['running_ts'],
                                                                var_length=1,
                                                                )
                with h5py.File(os.path.join(working_directory,'tone_tuning.h5'),'w') as f:
                    f.create_dataset('AMI', data=AMI)
                    f.create_dataset('p_value', data=p_value)
                    f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                    f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                    f.create_dataset('tuning_curves', data=tuning_curves)
        except:
            print('Could not extract tuning to single tone')
        
    elif data['task'] == 'legoSeqLT':
        try:
            if not os.path.exists(os.path.join(working_directory,'seqTone_tuning.h5')) or params['overwrite_mode']=='always':
                data = extract_seqLT_tone(data,params)
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_discrete_tuning(data['binaryData'],
                                                                data['seqLT_state'],
                                                                data['running_ts'],
                                                                var_length=3,
                                                                )
                
                with h5py.File(os.path.join(working_directory,'seqTone_tuning.h5'),'w') as f:
                    f.create_dataset('AMI', data=AMI)
                    f.create_dataset('p_value', data=p_value)
                    f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                    f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                    f.create_dataset('tuning_curves', data=tuning_curves)
        except:
            print('Could not extract tuning to tone sequence')