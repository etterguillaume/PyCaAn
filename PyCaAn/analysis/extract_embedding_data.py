#%% Import dependencies
import yaml
import numpy as np
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import tensorflow as tf
import os
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression as lin_reg
from PyCaAn.functions.decoding import decode_embedding
from PyCaAn.functions.signal_processing import extract_tone, extract_seqLT_tone
from PyCaAn.functions.dataloaders import load_data
from PyCaAn.functions.signal_processing import preprocess_data
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_embedding_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])
    if not os.path.exists(os.path.join(params['path_to_results'],'embedding_data')):
        os.mkdir(os.path.join(params['path_to_results'],'embedding_data'))

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        'embedding_data',
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    if not os.path.exists(os.path.join(working_directory,'model_params.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'model_params.h5'),'w') as f:

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
            embedding_model = ParametricUMAP(
                                verbose=False,
                                parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                                autoencoder_loss = True,
                                parametric_reconstruction = True,
                                n_components=params['embedding_dims'],
                                n_neighbors=params['n_neighbors'],
                                min_dist=params['min_dist'],
                                metric='euclidean',
                                random_state=42
                                ).fit(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])

            #train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
            train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
            test_embedding = embedding_model.transform(data['neuralData'][data['testingFrames'],0:params['input_neurons']])

            # Reconstruct inputs for both train and test sets
            reconstruction = embedding_model.inverse_transform(test_embedding)

            # Assess reconstruction error
            reconstruction_decoder = lin_reg().fit(reconstruction, data['rawData'][data['testingFrames']])
            reconstruction_score = reconstruction_decoder.score(reconstruction, data['rawData'][data['testingFrames']])

            # Save model data
            f.create_dataset('trainingFrames', data=data['trainingFrames'])
            f.create_dataset('testingFrames', data=data['testingFrames'])
            f.create_dataset('reconstruction_score', data=reconstruction_score)
        
        # Save model 
        # joblib.dump(embedding_model, os.path.join(working_directory,'model.sav')) # using joblib if non-parametric
        try:
            embedding_model.save(os.path.join(working_directory, 'model.h5'))
        except:
            print('Could not save parametric model')

    else: # Load existing model
    #     loaded_embedding_model = joblib.load(params['path_to_results'] + 'test_umap_model.sav') # ùsing joblib if non-parametric
        embedding_model = load_ParametricUMAP(working_directory, 'model.h5')
        model_file = h5py.File(os.path.join(working_directory,'model_params.h5'), 'r')
        data['trainingFrames'] = model_file['trainingFrames']
        data['testingFrames'] = model_file['testingFrames']
        reconstruction_score = model_file['reconstruction_score']
        model_file.close()

    #%% Decode
    # Decode elapsed time
    if not os.path.exists(os.path.join(working_directory,'retrospective_temporal_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['elapsed_time'],data, params, train_embedding, test_embedding)
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    if not os.path.exists(os.path.join(working_directory,'prospective_temporal_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_temporal_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['time2stop'],data, params, train_embedding, test_embedding)
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    # Decode distance travelled
    if not os.path.exists(os.path.join(working_directory,'retrospective_distance_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_distance_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['distance_travelled'],data, params, train_embedding, test_embedding)
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    if not os.path.exists(os.path.join(working_directory,'prospective_distance_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_distance_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['distance2stop'],data, params, train_embedding, test_embedding)
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
    
    # Decode velocity
    if not os.path.exists(os.path.join(working_directory,'velocity_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'velocity_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['velocity'],data, params, train_embedding, test_embedding)
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    # Decode position
    if not os.path.exists(os.path.join(working_directory,'spatial_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'spatial_decoding.h5'),'w') as f:
            if data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
                decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['position'],data, params, train_embedding, test_embedding)

            elif data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
                decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['position'][:,0],data, params, train_embedding, test_embedding)

            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    # Extract direction tuning
    try:
        if not os.path.exists(os.path.join(working_directory,'direction_decoding.h5')) or params['overwrite_mode']=='always':
            with h5py.File(os.path.join(working_directory,'direction_decoding.h5'),'w') as f:
                if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF':
                    decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['heading'],data, params, train_embedding, test_embedding)
                    f.create_dataset('decoding_error', data=decoding_error)
                    f.create_dataset('shuffled_error', data=shuffled_error)
                elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
                    decoding_score, z_score, p_value = decode_embedding(data['LT_direction'],data, params, train_embedding, test_embedding)
            
                f.create_dataset('decoding_score', data=decoding_score)
                f.create_dataset('z_score', data=z_score)
                f.create_dataset('p_value', data=p_value)

    except:
        print('Could not decode direction')

    # Decode tone
    if data['task'] == 'legoToneLT':
        try:
            if not os.path.exists(os.path.join(working_directory,'tone_decoding.h5')) or params['overwrite_mode']=='always':
                with h5py.File(os.path.join(working_directory,'tone_decoding.h5'),'w') as f:
                    data=extract_tone(data,params)
                    decoding_score, z_score, p_value = decode_embedding(data['binaryTone'],data, params, train_embedding, test_embedding)
            
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
        except:
            print('Could not decode single tone')
        
    elif data['task'] == 'legoSeqLT':
        try:
            if not os.path.exists(os.path.join(working_directory,'seqTone_decoding.h5')) or params['overwrite_mode']=='always':
                with h5py.File(os.path.join(working_directory,'seqTone_decoding.h5'),'w') as f:
                    data = extract_seqLT_tone(data,params)
                    decoding_score, z_score, p_value = decode_embedding(data['seqLT_state'],data, params, train_embedding, test_embedding)
            
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
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
    extract_embedding_session(data, params)