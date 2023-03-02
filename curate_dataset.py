#%% Imports
import yaml
import os
from tqdm import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.metrics import extract_total_distance_travelled

#%% Load YAML file
with open('params.yaml','r') as file:
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
            if folder.startswith('.'):
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
                else: #TODO add other conditions, like num cells, neurons, overwrite, etc
                    numFrames, numNeurons = data['rawData'].shape
                    distance_travelled=extract_total_distance_travelled(data['position'])
                    if numNeurons>=params['input_neurons'] and distance_travelled>=params['distance_travelled_threshold']:
                        path_list.append(session_path)
                    else:
                        excluded_list.append(session_path)

#%% Save list of sessions and stats in yaml files
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