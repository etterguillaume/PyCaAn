#%% Imports
import yaml
import os
import numpy as np
from argparse import ArgumentParser
from PyCaAn.functions.dataloaders import load_data
from PyCaAn.functions.signal_processing import preprocess_data, extract_tone, extract_seqLT_tone
from PyCaAn.functions.tuning import extract_tuning, extract_discrete_tuning
from PyCaAn.functions.metrics import extract_total_distance_travelled
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_tuning_session(data, params):
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

    # Extract tuning to time
    if not os.path.exists(os.path.join(working_directory,'retrospective_temporal_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_temporal_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_temporal_length']+params['temporalBinSize'],params['temporalBinSize']))
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(
                                                    data['binaryData'],
                                                    data['elapsed_time'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to prospective time
    if not os.path.exists(os.path.join(working_directory,'prospective_temporal_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_temporal_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_temporal_length']+params['temporalBinSize'],params['temporalBinSize']))
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(
                                                    data['binaryData'],
                                                    data['time2stop'],
                                                    data['running_ts'],
                                                    bins=bin_vec)

            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to retrospective distance
    if not os.path.exists(os.path.join(working_directory,'retrospective_distance_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_distance_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_distance_length']+params['spatialBinSize'],params['spatialBinSize']))
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                    data['distance_travelled'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to prospective distance
    if not os.path.exists(os.path.join(working_directory,'prospective_distance_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_distance_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_distance_length']+params['spatialBinSize'],params['spatialBinSize']))
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                    data['distance_travelled'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to velocity
    if not os.path.exists(os.path.join(working_directory,'velocity_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'velocity_tuning.h5'),'w') as f:
            bin_vec=(np.arange(params['speed_threshold'],params['max_velocity_length']+params['velocityBinSize'],params['velocityBinSize']))
            AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                    data['velocity'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('bins', data=bin_vec)

    # Extract spatial tuning
    if not os.path.exists(os.path.join(working_directory,'spatial_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'spatial_tuning.h5'),'w') as f:
            if data['task']=='OF':
                bin_vec=(np.arange(0,45+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,45+params['spatialBinSize'],params['spatialBinSize']))
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                                data['position'],
                                                                data['running_ts'],
                                                                bins=bin_vec)
                
            elif data['task']=='legoOF':
                bin_vec=(np.arange(0,50+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,50+params['spatialBinSize'],params['spatialBinSize']))
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                                data['position'],
                                                                data['running_ts'],
                                                                bins=bin_vec)
            elif data['task']=='plexiOF':
                bin_vec=(np.arange(0,49+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,49+params['spatialBinSize'],params['spatialBinSize']))
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                                data['position'],
                                                                data['running_ts'],
                                                                bins=bin_vec)

            elif data['task']=='LT':
                bin_vec=(np.arange(0,100+params['spatialBinSize'],params['spatialBinSize']))
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                                data['position'][:,0],
                                                                data['running_ts'],
                                                                bins=bin_vec)
                
            elif data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
                bin_vec=(np.arange(0,135+params['spatialBinSize'],params['spatialBinSize']))
                AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                                data['position'][:,0],
                                                                data['running_ts'],
                                                                bins=bin_vec)
            f.create_dataset('AMI', data=AMI)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('bins', data=bin_vec)

    # Extract direction tuning
    try:
        if not os.path.exists(os.path.join(working_directory,'direction_tuning.h5')) or params['overwrite_mode']=='always':
            with h5py.File(os.path.join(working_directory,'direction_tuning.h5'),'w') as f:
                if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF':
                    bin_vec=(np.arange(0,360+params['directionBinSize'],params['directionBinSize']))
                    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_tuning(data['binaryData'],
                                                                    data['heading'],
                                                                    data['running_ts'],
                                                                    bins=bin_vec)
                    f.create_dataset('bins', data=bin_vec)
                elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
                    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curves = extract_discrete_tuning(data['binaryData'],
                                                                    data['LT_direction'],
                                                                    data['running_ts'],
                                                                    var_length=1)
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
                with h5py.File(os.path.join(working_directory,'tone_tuning.h5'),'w') as f:
                    data=extract_tone(data,params)
                    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_discrete_tuning(data['binaryData'],
                                                                    data['binaryTone'],
                                                                    data['running_ts'],
                                                                    var_length=1,
                                                                    )                
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
                with h5py.File(os.path.join(working_directory,'seqTone_tuning.h5'),'w') as f:
                    data = extract_seqLT_tone(data,params)
                    AMI, p_value, occupancy_frames, active_frames_in_bin, tuning_curve = extract_discrete_tuning(data['binaryData'],
                                                                    data['seqLT_state'],
                                                                    data['running_ts'],
                                                                    var_length=3,
                                                                    )
                    f.create_dataset('AMI', data=AMI)
                    f.create_dataset('p_value', data=p_value)
                    f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                    f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                    f.create_dataset('tuning_curves', data=tuning_curves)
        except:
            print('Could not extract tuning to tone sequence')

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_tuning_session(data, params)