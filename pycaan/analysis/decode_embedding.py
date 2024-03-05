#%% Import dependencies
import yaml
import numpy as np
import os
from argparse import ArgumentParser
from pycaan.functions.decoding import decode_embedding, predict_embedding
from pycaan.functions.signal_processing import extract_tone, extract_seqLT_tone
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def decode_embedding_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )

    # Load embedding
    try:
        embedding_file = h5py.File(os.path.join(working_directory,'embedding.h5'),'r')
    except:
        print('Could not find embedding file. Please first extract embedding data.')
    # embedding = embedding_file['embedding'][()]
    train_embedding = embedding_file['train_embedding'][()]
    test_embedding = embedding_file['test_embedding'][()]
    data['trainingFrames'] = embedding_file['trainingFrames'][()]
    data['testingFrames'] = embedding_file['testingFrames'][()]
            
    # Decode
    # Decode elapsed time
    if not os.path.exists(os.path.join(working_directory,'retrospective_temporal_decoding.h5')) or params['overwrite_mode']=='always':
        decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['elapsed_time'],data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'retrospective_temporal_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
            f.create_dataset('test_prediction', data=test_prediction)

    if not os.path.exists(os.path.join(working_directory,'prospective_temporal_decoding.h5')) or params['overwrite_mode']=='always':
        decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['time2stop'],data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'prospective_temporal_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
            f.create_dataset('test_prediction', data=test_prediction)

    # Decode distance travelled
    if not os.path.exists(os.path.join(working_directory,'retrospective_distance_decoding.h5')) or params['overwrite_mode']=='always':
        decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['distance_travelled'],data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'retrospective_distance_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
            f.create_dataset('test_prediction', data=test_prediction)

    if not os.path.exists(os.path.join(working_directory,'prospective_distance_decoding.h5')) or params['overwrite_mode']=='always':
        decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['distance2stop'],data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'prospective_distance_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
            f.create_dataset('test_prediction', data=test_prediction)
    
    # Decode velocity
    if not os.path.exists(os.path.join(working_directory,'velocity_decoding.h5')) or params['overwrite_mode']=='always':
        decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['velocity'],data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'velocity_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
            f.create_dataset('test_prediction', data=test_prediction)

    # Decode position
    if not os.path.exists(os.path.join(working_directory,'spatial_decoding.h5')) or params['overwrite_mode']=='always':        
        if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF':
            decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['position'],data, params, train_embedding, test_embedding, isCircular=False)
        elif data['task'] == 'LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['position'][:,0],data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'spatial_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
            f.create_dataset('test_prediction', data=test_prediction)

    # Extract direction tuning
    if not os.path.exists(os.path.join(working_directory,'direction_decoding.h5')) or params['overwrite_mode']=='always':
        if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF' or data['task'] == 'smallOF':
            decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['heading'],data, params, train_embedding, test_embedding, isCircular=True)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
        elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
            decoding_score, z_score, p_value, _, _, test_prediction = decode_embedding(data['LT_direction'], data, params, train_embedding, test_embedding, isCircular=False)
        with h5py.File(os.path.join(working_directory,'direction_decoding.h5'),'w') as f:
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('test_prediction', data=test_prediction)

    # Decode tone
    if data['task'] == 'legoToneLT':
        if not os.path.exists(os.path.join(working_directory,'tone_decoding.h5')) or params['overwrite_mode']=='always':
            data=extract_tone(data,params)
            decoding_score, z_score, p_value, _, _, test_prediction = decode_embedding(data['binaryTone'],data, params, train_embedding, test_embedding, isCircular=False)
            with h5py.File(os.path.join(working_directory,'tone_decoding.h5'),'w') as f:
                f.create_dataset('decoding_score', data=decoding_score)
                f.create_dataset('z_score', data=z_score)
                f.create_dataset('p_value', data=p_value)
                f.create_dataset('test_prediction', data=test_prediction)
        
    elif data['task'] == 'legoSeqLT':
        if not os.path.exists(os.path.join(working_directory,'seqTone_decoding.h5')) or params['overwrite_mode']=='always':
            data = extract_seqLT_tone(data,params)
            decoding_score, z_score, p_value, _, _, test_prediction = decode_embedding(data['seqLT_state'],data, params, train_embedding, test_embedding, isCircular=False)
            with h5py.File(os.path.join(working_directory,'seqTone_decoding.h5'),'w') as f:
                f.create_dataset('decoding_score', data=decoding_score)
                f.create_dataset('z_score', data=z_score)
                f.create_dataset('p_value', data=p_value)
                f.create_dataset('test_prediction', data=test_prediction)

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    decode_embedding_session(data, params)