#%% Import dependencies
import yaml
import numpy as np
from umap.umap_ import UMAP
import joblib
import os
import sys
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression as lin_reg
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_neural_structure_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    if not os.path.exists(os.path.join(working_directory,'neural_structure.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'neural_structure.h5'),'w') as f:

            # Split dataset
            trainingFrames = np.zeros(len(data['caTime']), dtype=bool)

            if params['train_set_selection']=='random':
                trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
            elif params['train_set_selection']=='split':
                trainingFrames[0:int(params['train_test_ratio']*len(data['caTime']))] = True 

            data['trainingFrames'] = trainingFrames
            data['testingFrames'] = ~trainingFrames

            # Exclude immobility from all sets
            data['trainingFrames'][~data['running_ts']] = False
            data['testingFrames'][~data['running_ts']] = False

            # Train neural_structure model

            neural_structure_model = UMAP(
                        n_components=2, # project on 2D
                        n_neighbors=params['n_neighbors'],
                        min_dist=params['min_dist'],
                        metric='euclidean',
                        random_state=params['seed']
                        ).fit(data['neuralData'][data['trainingFrames'],0:params['input_neurons']].T)

            train_neural_structure = neural_structure_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']].T)
            test_neural_structure = neural_structure_model.transform(data['neuralData'][data['testingFrames'],0:params['input_neurons']].T)
            neural_structure = neural_structure_model.transform(data['neuralData'][:,0:params['input_neurons']].T)

            # Assess reconstruction error
            stability_decoder = lin_reg().fit(test_neural_structure, train_neural_structure)
            stability_score = stability_decoder.score(test_neural_structure, train_neural_structure)

            # Save embedding data
            f.create_dataset('trainingFrames', data=data['trainingFrames'])
            f.create_dataset('testingFrames', data=data['testingFrames'])
            f.create_dataset('reconstruction_score', data=stability_score)
            f.create_dataset('train_embedding', data=train_neural_structure)
            f.create_dataset('test_embedding', data=test_neural_structure)
            f.create_dataset('neural_structure', data=neural_structure)

            # joblib.dump(neural_structure_model, os.path.join(working_directory,'neural_structure_model.sav')) # no need to save

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_neural_structure_session(data, params)