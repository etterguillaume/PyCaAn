#%% Imports
import yaml
import os
import numpy as np
from argparse import ArgumentParser
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.tuning import extract_tuning
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
    if not os.path.exists(os.path.join(working_directory,'internal_tuning.h5')) or params['overwrite_mode']=='always':
        
        # Load embedding
        # Normalize embedding?


        bin_vec = (np.arange(0,params['max_temporal_length']+params['temporalBinSize'],params['temporalBinSize']))
        info, p_value, occupancy_frames, active_frames_in_bin, tuning_curves, peak_loc, peak_val = extract_tuning(
                                                data['binaryData'],
                                                data['elapsed_time'],
                                                data['running_ts'],
                                                bins=bin_vec)
        
        with h5py.File(os.path.join(working_directory,'internal_tuning.h5'),'w') as f:
            f.create_dataset('info', data=info)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('occupancy_frames', data=occupancy_frames, dtype=int)
            f.create_dataset('active_frames_in_bin', data=active_frames_in_bin, dtype=int)
            f.create_dataset('tuning_curves', data=tuning_curves)
            f.create_dataset('peak_loc', data=peak_loc)
            f.create_dataset('peak_val', data=peak_val)
            f.create_dataset('bins', data=bin_vec)

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_tuning_session(data, params)