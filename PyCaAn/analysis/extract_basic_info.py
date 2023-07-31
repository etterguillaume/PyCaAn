#%% Import dependencies
import yaml
import os
import h5py
from argparse import ArgumentParser
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.metrics import extract_total_distance_travelled, extract_firing_properties

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_basic_info_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    # Save basic info
    numFrames, numNeurons = data['rawData'].shape
    total_distance_travelled = extract_total_distance_travelled(data['position'])
    marginal_likelihood, trans_prob = extract_firing_properties(data['binaryData'])

    info_dict = {
                'path': data['path'],
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
                'duration': float(data['caTime'][-1]),
        }
    if not os.path.exists(os.path.join(working_directory,'info.yaml')) or params['overwrite_mode']=='always':
        with open(os.path.join(working_directory,'info.yaml'),"w") as file:
            yaml.dump(info_dict,file)

    if not os.path.exists(os.path.join(working_directory,'data_stats.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'data_stats.h5'),"w") as f:
            f.create_dataset('marginal_likelihood', data=marginal_likelihood)
            f.create_dataset('trans_prob', data=trans_prob)
            f.create_dataset('total_distance_travelled', data=float(total_distance_travelled))

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_basic_info_session(data, params)