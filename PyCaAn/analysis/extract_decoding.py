#%% Imports
import yaml
import numpy as np
import os
from tqdm import tqdm
from PyCaAn.functions.decoding import decode_neural_data
from PyCaAn.functions.dataloaders import load_data
from PyCaAn.functions.signal_processing import preprocess_data
from PyCaAn.functions.signal_processing import extract_tone, extract_seqLT_tone
import h5py

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Load folders to analyze from yaml file?
with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
    session_file = yaml.full_load(file)
session_list = session_file['sessions']
print(f'{len(session_list)} sessions to process')

#%% If tuning_data folder does not exist, create it
if not os.path.exists(params['path_to_results']):
    os.mkdir(params['path_to_results'])
if not os.path.exists(os.path.join(params['path_to_results'],'decoding_data')):
    os.mkdir(os.path.join(params['path_to_results'],'decoding_data'))

for i, session in enumerate(tqdm(session_list)):
    data = load_data(session)

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        'embedding_data',
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    # Preprocessing 
    data = preprocess_data(data,params)

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

    #%% Decode
    # Decode elapsed time
    if not os.path.exists(os.path.join(working_directory,'retrospective_temporal_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_decoding.h5'),'w') as f:
            
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['elapsed_time'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            #decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_embedding(data['elapsed_time'],data, params, train_embedding, test_embedding)
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    if not os.path.exists(os.path.join(working_directory,'prospective_temporal_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_temporal_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['time2stop'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    # Decode distance travelled
    if not os.path.exists(os.path.join(working_directory,'retrospective_distance_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'retrospective_distance_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['distance_travelled'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    if not os.path.exists(os.path.join(working_directory,'prospective_distance_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'prospective_distance_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['distance2stop'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)
    
    # Decode velocity
    if not os.path.exists(os.path.join(working_directory,'velocity_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'velocity_decoding.h5'),'w') as f:
            decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['velocity'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
            f.create_dataset('decoding_error', data=decoding_error)
            f.create_dataset('shuffled_error', data=shuffled_error)

    # Decode position
    if not os.path.exists(os.path.join(working_directory,'spatial_decoding.h5')) or params['overwrite_mode']=='always':
        with h5py.File(os.path.join(working_directory,'spatial_decoding.h5'),'w') as f:
            if data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
                decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['position'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])

            elif data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
                decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['position'][:,0],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])

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
                    decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['heading'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
                    f.create_dataset('decoding_error', data=decoding_error)
                    f.create_dataset('shuffled_error', data=shuffled_error)
                elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
                    decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['LT_direction'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            
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
                    decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['binaryTone'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            
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
                    decoding_score, z_score, p_value, decoding_error, shuffled_error = decode_neural_data(data['seqLT_state'],
                                                                                                  data['binaryData'][:,0:params['input_neurons']],
                                                                                                  params,
                                                                                                  data['trainingFrames'],
                                                                                                  data['testingFrames'])
            
            f.create_dataset('decoding_score', data=decoding_score)
            f.create_dataset('z_score', data=z_score)
            f.create_dataset('p_value', data=p_value)
        except:
            print('Could not extract tuning to tone sequence')