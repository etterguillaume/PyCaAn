#%% Import dependencies
import yaml
import numpy as np
import os
from argparse import ArgumentParser
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.simulate import model_data
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_model_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    if not os.path.exists(os.path.join(working_directory,'model_data.h5')) or params['overwrite_mode']=='always':
        modeled_place_activity, modeled_grid_activity, modeled_BVC_activity = model_data(data, params)
        
        with h5py.File(os.path.join(working_directory,'model_data.h5'),'w') as f:
            f.create_dataset('modeled_place_activity', data=modeled_place_activity)
            f.create_dataset('modeled_grid_activity', data=modeled_grid_activity)
            f.create_dataset('modeled_BVC_activity', data=modeled_BVC_activity)   

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_model_session(data, params)