#%% Imports
import yaml
import os
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import extract_1D_tuning, extract_2D_tuning
from functions.metrics import extract_total_distance_travelled
import h5py

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open('batchList.yaml','r') as file:
    session_file = yaml.full_load(file)
session_list = session_file['sessions']
print(f'{len(session_list)} sessions to process')
#session_list=['../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621']

#%%
for i, session in enumerate(tqdm(session_list)):
    data = load_data(session)

    # If region folder does not exist, create it
    if not os.path.exists(os.path.join('output',data['region'])): # If folder does not exist, create it
        os.mkdir(os.path.join('output',data['region']))

    # If subject folder does not exist, create it
    if not os.path.exists(os.path.join('output',data['region'],data['subject'])): # If folder does not exist, create it
        os.mkdir(os.path.join('output',data['region'],data['subject']))
    
    working_directory = os.path.join('output',data['region'],data['subject'],str(data['day']))
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

    #TODO add bin parameters here depending on task

    with open(os.path.join(working_directory,'info.yaml'),"w") as file:
        yaml.dump(info_dict,file)

    # Extract tuning to time
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['elapsed_time'],
                                                        data['running_ts'],
                                                        var_length=params['max_temporal_length'],
                                                        bin_size=params['temporalBinSize'])
    data_dict={
        'AMI':AMI,
        'occupancy_frames': occupancy_frames,
        'active_frames_in_bin': active_frames_in_bin,
        'tuning_curves': tuning_curves
    }
    with h5py.File(os.path.join(working_directory,'temporal_tuning.h5'),'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)

    # Extract tuning to distance
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['distance_travelled'],
                                                        data['running_ts'],
                                                        var_length=params['max_distance_length'],
                                                        bin_size=params['spatialBinSize'])

    data_dict={
        'AMI':AMI,
        'occupancy_frames': occupancy_frames,
        'active_frames_in_bin': active_frames_in_bin,
        'tuning_curves': tuning_curves
    }
    with h5py.File(os.path.join(working_directory,'distance_tuning.h5'),'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)

    # Extract tuning to velocity
    AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
                                                        data['velocity'],
                                                        data['running_ts'],
                                                        var_length=params['max_velocity_length'],
                                                        bin_size=params['velocityBinSize'])

    data_dict={
        'AMI':AMI,
        'occupancy_frames': occupancy_frames,
        'active_frames_in_bin': active_frames_in_bin,
        'tuning_curves': tuning_curves
    }
    with h5py.File(os.path.join(working_directory,'velocity_tuning.h5'),'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)

    # Extract spatial tuning
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
        
    data_dict={
        'AMI':AMI,
        'occupancy_frames': occupancy_frames,
        'active_frames_in_bin': active_frames_in_bin,
        'tuning_curves': tuning_curves
    }
    with h5py.File(os.path.join(working_directory,'spatial_tuning.h5'),'w') as f:
        for k, v in data_dict.items():
            f.create_dataset(k, data=v)

    # Extract direction tuning
    # if data['task']=='legoOF' or data['task']=='OF':
    #     HD=[] # TODO extract head direction tuning?

    # if data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
    #     AMI, occupancy_frames, active_frames_in_bin, tuning_curves = extract_1D_tuning(data['binaryData'],
    #                                                     data['LT_direction'],
    #                                                     data['running_ts'],
    #                                                     var_length=2,
    #                                                     bin_size=1)

    # data_dict={
    #     'AMI':AMI,
    #     'occupancy_frames': occupancy_frames,
    #     'active_frames_in_bin': active_frames_in_bin,
    #     'tuning_curves': tuning_curves
    # }
    # with h5py.File(os.path.join(working_directory,'direction_tuning.h5'),'w') as f:
    #     for k, v in data_dict.items():
    #         f.create_dataset(k, data=v)

    # Extract tuning to tone
    # if data['task'] == 'legoToneLT':
    #     AMI, occupancy, tuning_curves = extract_1D_tuning(data['binaryData'],
    #                                                     data['tone'],
    #                                                     data['running_ts'],
    #                                                     var_length=#TODO,
    #                                                     bin_size=#TODO
    #                                                     )
        
    # if data['task'] == 'legoSeqLT':
    #     AMI, occupancy, tuning_curves = extract_1D_tuning(data['binaryData'],
    #                                                     data['tone'],
    #                                                     data['running_ts'],
    #                                                     var_length=4,
    #                                                     bin_size=1
    #                                                     )

    # TODO check if folder exists, check if params['overwrite'] is 'changes_only', 'always' or 'never'
    # TODO if exists, check results contents. If params_old==params_new, decide whether overwrite
    
    # Update batchfile to remove processed file

    session_list.pop(i)
    sessions_dict = {'sessions':session_list}
    with open('batchList.yaml','w') as file:
        yaml.dump(sessions_dict,file)
    
# %%
