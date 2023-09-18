#%% Imports
import yaml
import os
import numpy as np
from argparse import ArgumentParser
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data, extract_tone, extract_seqLT_tone
from pycaan.functions.tuning import extract_tuning, extract_discrete_tuning
from pycaan.functions.metrics import extract_total_distance_travelled
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_tuning_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    # Extract tuning to time
    if not os.path.exists(os.path.join(working_directory,'retrospective_temporal_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_temporal_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_temporal_length']+params['temporalBinSize'],params['temporalBinSize']))
            info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(
                                                    data['binaryData'],
                                                    data['elapsed_time'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to prospective time
    if not os.path.exists(os.path.join(working_directory,'prospective_temporal_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_temporal_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_temporal_length']+params['temporalBinSize'],params['temporalBinSize']))
            info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(
                                                    data['binaryData'],
                                                    data['time2stop'],
                                                    data['running_ts'],
                                                    bins=bin_vec)

            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to retrospective distance
    if not os.path.exists(os.path.join(working_directory,'retrospective_distance_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_distance_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_distance_length']+params['distanceBinSize'],params['distanceBinSize']))
            info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                    data['distance_travelled'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to prospective distance
    if not os.path.exists(os.path.join(working_directory,'prospective_distance_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_distance_tuning.h5'),'w') as f:
            bin_vec = (np.arange(0,params['max_distance_length']+params['distanceBinSize'],params['distanceBinSize']))
            info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                    data['distance2stop'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to velocity
    if not os.path.exists(os.path.join(working_directory,'velocity_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'velocity_tuning.h5'),'w') as f:
            bin_vec=(np.arange(params['speed_threshold'],params['max_velocity_length']+params['velocityBinSize'],params['velocityBinSize']))
            info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                    data['velocity'],
                                                    data['running_ts'],
                                                    bins=bin_vec)
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

    # Extract tuning to acceleration
    if not os.path.exists(os.path.join(working_directory,'acceleration_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'acceleration_tuning.h5'),'w') as f:
            info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_discrete_tuning(data['binaryData'],
                                                                    data['acceleration'],
                                                                    data['running_ts'],
                                                                    )
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)

    # Extract spatial tuning
    if not os.path.exists(os.path.join(working_directory,'spatial_tuning.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'spatial_tuning.h5'),'w') as f:
            if data['task']=='OF':
                bin_vec=(np.arange(0,45+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,45+params['spatialBinSize'],params['spatialBinSize']))
                info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                                data['position'],
                                                                data['running_ts'],
                                                                bins=bin_vec)
                
            elif data['task']=='legoOF':
                bin_vec=(np.arange(0,50+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,50+params['spatialBinSize'],params['spatialBinSize']))
                info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                                data['position'],
                                                                data['running_ts'],
                                                                bins=bin_vec)
            elif data['task']=='plexiOF':
                bin_vec=(np.arange(0,49+params['spatialBinSize'],params['spatialBinSize']),
                         np.arange(0,49+params['spatialBinSize'],params['spatialBinSize']))
                info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                                data['position'],
                                                                data['running_ts'],
                                                                bins=bin_vec)

            elif data['task']=='LT':
                bin_vec=(np.arange(0,100+params['spatialBinSize'],params['spatialBinSize']))
                info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                                data['position'][:,0],
                                                                data['running_ts'],
                                                                bins=bin_vec)
                
            elif data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
                bin_vec=(np.arange(0,135+params['spatialBinSize'],params['spatialBinSize']))
                info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                                data['position'][:,0],
                                                                data['running_ts'],
                                                                bins=bin_vec)
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

    # Extract direction tuning
    try:
        if not os.path.exists(os.path.join(working_directory,'direction_tuning.h5')) or params['overwrite_mode']=='always':
            with h5py.File(os.path.join(working_directory,'direction_tuning.h5'),'w') as f:
                if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF':
                    bin_vec=(np.arange(0,360+params['directionBinSize'],params['directionBinSize']))
                    info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(data['binaryData'],
                                                                    data['heading'],
                                                                    data['running_ts'],
                                                                    bins=bin_vec)
                    f.create_dataset('bins', data=bin_vec)
                elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
                    info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_discrete_tuning(data['binaryData'],
                                                                    data['LT_direction'],
                                                                    data['running_ts'])
                f.create_dataset('info', data=info)
                f.create_dataset('p_value', data=p_value)
                f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                f.create_dataset('tuning_curves', data=tuning_curves)
                f.create_dataset('peak_loc', data=peak_loc)
                f.create_dataset('peak_val', data=peak_val)

    except:
        print('Could not extract tuning to direction')

    # Extract tuning to tone
    if data['task'] == 'legoToneLT':
        try:
            if not os.path.exists(os.path.join(working_directory,'tone_tuning.h5')) or params['overwrite_mode']=='always':
                with h5py.File(os.path.join(working_directory,'tone_tuning.h5'),'w') as f:
                    data=extract_tone(data,params)
                    info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_discrete_tuning(data['binaryData'],
                                                                    data['binaryTone'],
                                                                    data['running_ts'],
                                                                    )                
                    f.create_dataset('info', data=info)
                    f.create_dataset('p_value', data=p_value)
                    f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                    f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                    f.create_dataset('tuning_curves', data=tuning_curves)
                    f.create_dataset('peak_loc', data=peak_loc)
                    f.create_dataset('peak_val', data=peak_val)
        except:
            print('Could not extract tuning to single tone')
        
    elif data['task'] == 'legoSeqLT':
        try:
            if not os.path.exists(os.path.join(working_directory,'seqTone_tuning.h5')) or params['overwrite_mode']=='always':
                with h5py.File(os.path.join(working_directory,'seqTone_tuning.h5'),'w') as f:
                    data = extract_seqLT_tone(data,params)
                    info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_discrete_tuning(data['binaryData'],
                                                                    data['seqLT_state'],
                                                                    data['running_ts'],
                                                                    )
                    f.create_dataset('info', data=info)
                    f.create_dataset('p_value', data=p_value)
                    f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
                    f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
                    f.create_dataset('tuning_curves', data=tuning_curves)
                    f.create_dataset('peak_loc', data=peak_loc)
                    f.create_dataset('peak_val', data=peak_val)
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