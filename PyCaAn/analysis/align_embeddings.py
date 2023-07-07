#%% Imports
import numpy as np
import h5py
import os
import yaml
from tqdm import tqdm
import itertools
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.embedding import extract_hyperalignment_score
import pandas as pd

def align_embeddings(params):
    #%% List sessions
    sessionList=os.listdir(os.path.join(params['path_to_results'],'results'))
    #for i,session in enumerate(sessionList):
    try:
        sessionList.remove('.DS_Store') #TODO generalize garbage removal
    except:
        print('')

    #%% Initialize matrices
    data_list = []

    #%% Extract data
    for session_A, session_B in tqdm(list(itertools.product(sessionList,sessionList)), total=len(sessionList)**2):
        try:
            info_file_A=open(os.path.join(params['path_to_results'],'results',session_A,'info.yaml'),'r')
            session_A_info = yaml.full_load(info_file_A)
            info_file_B=open(os.path.join(params['path_to_results'],'results',session_B,'info.yaml'),'r')
            session_B_info = yaml.full_load(info_file_B)

            if session_A_info['task']==session_B_info['task']: # Only compare manifolds on similar tasks
                task=session_A_info['task']
                if task=='legoLT' or task=='legoToneLT' or task=='legoSeqLT':
                    if session_A_info['path']==session_B_info['path']:
                        sameSession=True
                    else:
                        sameSession=False
                    if session_A_info['subject']==session_B_info['subject']:
                        sameMouse=True
                        sameRegion=True
                    elif session_A_info['region']==session_B_info['region']:
                        sameMouse=False
                        sameRegion=True
                    else:
                        sameMouse=False
                        sameRegion=False

                    data_A = preprocess_data(load_data(session_A_info['path']), params)
                    data_B = preprocess_data(load_data(session_B_info['path']), params)
                    with h5py.File(os.path.join(params['path_to_results'],'results',session_A,'embedding.h5'), 'r') as f:
                        embedding_A=f['embedding'][()]
                        trainingFrames_A=f['trainingFrames'][()]
                        testingFrames_A=f['testingFrames'][()]
                        reconstruction_score_A=f['reconstruction_score'][()]
                    with h5py.File(os.path.join(params['path_to_results'],'results',session_B,'embedding.h5'), 'r') as f:
                        embedding_B=f['embedding'][()]
                        trainingFrames_B=f['trainingFrames'][()]
                        testingFrames_B=f['testingFrames'][()]
                        reconstruction_score_B=f['reconstruction_score'][()]
                
                    ## Hyperalignment
                    # Time
                    bin_vec=(np.linspace(0,params['max_temporal_length'],params['quantization_steps']))
                    temporal_HAS_AB, temporal_HAS_BA, _, _, _, _, _ = extract_hyperalignment_score(
                        embedding_ref=embedding_A,
                        var_ref=data_A['elapsed_time'],
                        trainingFrames_ref = trainingFrames_A,
                        testingFrames_ref = testingFrames_A,
                        embedding_pred=embedding_B,
                        var_pred=data_B['elapsed_time'],
                        trainingFrames_pred = trainingFrames_B,
                        testingFrames_pred = testingFrames_B,
                        bin_vec=bin_vec
                        )

                    # Distance
                    bin_vec=(np.linspace(0,params['max_distance_length'],params['quantization_steps']))
                    distance_HAS_AB, distance_HAS_BA, _, _, _, _, _ = extract_hyperalignment_score(
                        embedding_ref=embedding_A,
                        var_ref=data_A['distance_travelled'],
                        trainingFrames_ref = trainingFrames_A,
                        testingFrames_ref = testingFrames_A,
                        embedding_pred=embedding_B,
                        var_pred=data_B['distance_travelled'],
                        trainingFrames_pred = trainingFrames_B,
                        testingFrames_pred = testingFrames_B,
                        bin_vec=bin_vec
                        )
                    
                    # Speed
                    bin_vec=(np.linspace(0,params['max_velocity_length'],params['quantization_steps']))
                    speed_HAS_AB, speed_HAS_BA, _, _, _, _, _ = extract_hyperalignment_score(
                        embedding_ref=embedding_A,
                        var_ref=data_A['velocity'],
                        trainingFrames_ref = trainingFrames_A,
                        testingFrames_ref = testingFrames_A,
                        embedding_pred=embedding_B,
                        var_pred=data_B['velocity'],
                        trainingFrames_pred = trainingFrames_B,
                        testingFrames_pred = testingFrames_B,
                        bin_vec=bin_vec
                        )
                    
                    # Space
                    if task=='LT':
                        bin_vec=(np.linspace(0,100,params['quantization_steps']))
                    elif task=='legoLT' or task=='legoToneLT' or task=='legoSeqLT':
                        bin_vec=(np.linspace(0,134,params['quantization_steps']))
                    spatial_HAS_AB, spatial_HAS_BA, _, _, _, _, _ = extract_hyperalignment_score(
                        embedding_ref=embedding_A,
                        var_ref=data_A['position'][:,0],
                        trainingFrames_ref = trainingFrames_A,
                        testingFrames_ref = testingFrames_A,
                        embedding_pred=embedding_B,
                        var_pred=data_B['position'][:,0],
                        trainingFrames_pred = trainingFrames_B,
                        testingFrames_pred = testingFrames_B,
                        bin_vec=bin_vec
                        )

                    data_list.append(
                    {   
                        'Task': task,
                        'Reference_region': session_A_info['region'],
                        'Reference_subject': session_A_info['subject'],
                        'Reference_day': session_A_info['day'],
                        'Predicted_region': session_B_info['region'],
                        'Predicted_subject': session_B_info['subject'],
                        'Predicted_day': session_B_info['day'],
                        'sameSession': sameSession,
                        'sameMouse': sameMouse,
                        'sameRegion': sameRegion,
                        'reconstruction_score_ref': reconstruction_score_A,
                        'reconstruction_score_pred': reconstruction_score_B,
                        'temporal_HAS': temporal_HAS_AB,
                        'distance_HAS': distance_HAS_AB,
                        'speed_HAS': speed_HAS_AB,
                        'spatial_HAS': spatial_HAS_AB,
                        }
                    )

                    data_list.append(
                    {
                        'Task': task,
                        'Reference_region': session_B_info['region'],
                        'Reference_subject': session_B_info['subject'],
                        'Reference_day': session_B_info['day'],
                        'Predicted_region': session_A_info['region'],
                        'Predicted_subject': session_A_info['subject'],
                        'Predicted_day': session_A_info['day'],
                        'sameSession': sameSession,
                        'sameMouse': sameMouse,
                        'sameRegion': sameRegion,
                        'reconstruction_score_ref': reconstruction_score_B,
                        'reconstruction_score_pred': reconstruction_score_A,
                        'temporal_HAS': temporal_HAS_BA,
                        'distance_HAS': distance_HAS_BA,
                        'speed_HAS': speed_HAS_BA,
                        'spatial_HAS': spatial_HAS_BA,
                        }
                    )
                    
        except:
            print(f'Could not open sessions {session_A} or {session_B}')

    df = pd.DataFrame(data_list)
    df.to_csv(os.path.join(params['path_to_results'], 'hyperalignment_data.csv'))

if __name__ == '__main__': 
    with open('params.yaml','r') as file:
        params = yaml.full_load(file)
    align_embeddings(params)