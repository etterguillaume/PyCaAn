#%% Import dependencies
import yaml
import numpy as np
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from umap.umap_ import UMAP
import tensorflow as tf
import joblib
import os
import sys
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression as lin_reg
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
import h5py

class hide_output_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_embedding_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])
    if not os.path.exists(os.path.join(params['path_to_results'], 'results')):
        os.mkdir(os.path.join(params['path_to_results'], 'results'))

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        'results',
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    if not os.path.exists(os.path.join(working_directory,'embedding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'embedding.h5'),'w') as f:

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

            # Train embedding model
            if params['parametric_embedding']:
                with hide_output_prints():
                    embedding_model = ParametricUMAP(
                                verbose=False,
                                parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                                autoencoder_loss = True,
                                parametric_reconstruction = True,
                                n_components=params['embedding_dims'],
                                n_neighbors=params['n_neighbors'],
                                min_dist=params['min_dist'],
                                metric='euclidean',
                                random_state=params['seed']
                                ).fit(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
            else:
                embedding_model = UMAP(
                                n_components=params['embedding_dims'],
                                n_neighbors=params['n_neighbors'],
                                min_dist=params['min_dist'],
                                metric='euclidean',
                                random_state=params['seed']
                                ).fit(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])

            #train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
            train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
            test_embedding = embedding_model.transform(data['neuralData'][data['testingFrames'],0:params['input_neurons']])
            embedding = embedding_model.transform(data['neuralData'][:,0:params['input_neurons']])

            # Reconstruct inputs for both train and test sets
            reconstruction = embedding_model.inverse_transform(test_embedding)

            # Assess reconstruction error
            reconstruction_decoder = lin_reg().fit(reconstruction, data['rawData'][data['testingFrames']])
            reconstruction_score = reconstruction_decoder.score(reconstruction, data['rawData'][data['testingFrames']])

            # Save embedding data
            f.create_dataset('trainingFrames', data=data['trainingFrames'])
            f.create_dataset('testingFrames', data=data['testingFrames'])
            f.create_dataset('reconstruction_score', data=reconstruction_score)
            f.create_dataset('train_embedding', data=train_embedding)
            f.create_dataset('test_embedding', data=test_embedding)
            f.create_dataset('embedding', data=embedding)

        # Save model 
        joblib.dump(embedding_model, os.path.join(working_directory,'model.sav')) # using joblib if non-parametric
            
    # else: # Load existing model
    # #     loaded_embedding_model = joblib.load(params['path_to_results'] + 'test_umap_model.sav') # ùsing joblib if non-parametric


# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_embedding_session(data, params)