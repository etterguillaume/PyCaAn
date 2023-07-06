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

#%% Params
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% List sessions
sessionList=os.listdir(os.path.join(params['path_to_results'],'results'))
for i,session in enumerate(sessionList):
    if '_LT_' not in session:
        sessionList.pop(i)

sessionList.remove('.DS_Store') #TODO generalize garbage removal

#%% Initialize matrices
data_list = []

#%% Extarct data
for session_A, session_B in tqdm(list(itertools.product(sessionList,sessionList)), total=len(sessionList)**2):
    try:
        info_file_A=open(os.path.join(params['path_to_results'],'results',session_A,'info.yaml'),'r')
        session_A_info = yaml.full_load(info_file_A)
        info_file_B=open(os.path.join(params['path_to_results'],'results',session_B,'info.yaml'),'r')
        session_B_info = yaml.full_load(info_file_B)
    except:
        print(f'Could not open sessions {session_A} or {session_B}')

    if session_A_info['task']==session_B_info['task']:
        task=session_A_info['task']
        if task=='LT': #TODO temporary
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
        
            # Hyperalignment here
            if task=='LT':
                bin_vec=(np.arange(0,100+params['spatialBinSize'],params['spatialBinSize']*2))
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
                'reconstruction_score_A': reconstruction_score_A,
                'reconstruction_score_B': reconstruction_score_B,
                'spatial_HAS': spatial_HAS_AB
                }
            )

            data_list.append(
            {
                'Task':task,
                'Reference_session': session_B_info['region'] + session_B_info['subject'] + str(session_B_info['day']),
                'Predicted_session': session_A_info['region'] + session_A_info['subject'] + str(session_A_info['day']),
                'sameMouse': sameMouse,
                'sameRegion': sameRegion,
                'reconstruction_score_A': reconstruction_score_A,
                'reconstruction_score_B': reconstruction_score_B,
                'spatial_HAS': spatial_HAS_BA
                }
            )

#%% 
df = pd.DataFrame(data_list)

#%% Save dataframe
df.to_csv(os.path.join(params['path_to_results'], 'results','hyperalignment_data.csv'))