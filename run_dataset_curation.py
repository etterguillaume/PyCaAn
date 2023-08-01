#%% Imports
import yaml
import os
from tqdm import tqdm
import numpy as np
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.metrics import extract_total_distance_travelled
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    parser.add_argument('--param_file', type=str, default='params.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments() #TODO add params override here
    config = vars(args)

    with open(args.param_file,'r') as file:
        params = yaml.full_load(file)

    #%% Establish list of regions
    path_list = []
    error_list = []
    excluded_list = []
    regionList = os.listdir(params['path_to_dataset'])
    for i, folder in enumerate(regionList):
        if folder.startswith('.'):
            regionList.pop(i)
    numRegions = len(regionList)
    numSubjects={}
    numSessions={}
    ct=0
    for region in regionList:
        subjectList=os.listdir(os.path.join(params['path_to_dataset'],region))
        for i, folder in enumerate(subjectList):
            if folder.startswith('.'):
                subjectList.pop(i)
        numSubjects.update({region:len(subjectList)})
        for subject in tqdm(subjectList):
            sessionList=os.listdir(os.path.join(params['path_to_dataset'],region, subject))
            for i, folder in enumerate(sessionList):
                if folder.startswith('.') or not folder.endswith('.mat'):
                    sessionList.pop(i)

            numSessions.update({subject:len(sessionList)})
            for session in sessionList:
                session_path = os.path.join(params['path_to_dataset'],region,subject,session)
                if os.path.isfile(os.path.join(session_path,'ms.mat')) and os.path.isfile(os.path.join(session_path,'behav.mat')):
                    try:
                        data = load_data(session_path)
                        data = preprocess_data(data, params)
                    except:
                        error_list.append(session_path)
                        print(f'Could not open {session_path}')
                    else:
                        path_list.append(session_path)
                        numFrames, numNeurons = data['rawData'].shape
                        distance_travelled=extract_total_distance_travelled(data['position'])
                        if numNeurons<params['input_neurons'] or distance_travelled<params['distance_travelled_threshold']:
                            excluded_list.append(session_path)

    #%% Save list of sessions and stats in yaml files
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    sessions_dict = {'sessions':path_list}
    with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'w') as file:
        yaml.dump(sessions_dict,file)

    excluded_dict = {'sessions':excluded_list}
    with open(os.path.join(params['path_to_results'],'excludedList.yaml'),'w') as file:
        yaml.dump(excluded_dict,file)

    # Save files that could not be opened for reference
    error_file_dict = {'sessions':error_list}
    with open(os.path.join(params['path_to_results'],'errorFileList.yaml'),'w') as file:
        yaml.dump(error_file_dict,file)
                    
    dataset_stats_dict = {'numRegions': numRegions, 
                        'numSubjects': numSubjects, 
                        'numSessions': numSessions}

    with open(os.path.join(params['path_to_results'],'dataset_stats.yaml'),'w') as file:
        yaml.dump(dataset_stats_dict,file)

    # Track current parameters
    if not os.path.exists(os.path.join(params['path_to_results'],'params.yaml')) or params['overwrite_mode']=='always':
        with open(os.path.join(params['path_to_results'],'params.yaml'),"w") as file:
            yaml.dump(params,file)