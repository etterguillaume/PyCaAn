#%%
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
import tqdm
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data

#%% Load YAML file
with open('params.yaml','r') as file:
    params = yaml.full_load(file)
#%% Establish list of regions
sessionList = []
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
    for subject in subjectList:
        sessionList=os.listdir(os.path.join(params['path_to_dataset'],region, subject))
        for i, folder in enumerate(sessionList):
            if folder.startswith('.'):
                sessionList.pop(i)
        numSessions.update({subject:len(sessionList)})
        for session in sessionList:
            session_path = os.path.join(params['path_to_dataset'],region,subject,session)
            if os.path.isfile(os.path.join(session_path,'ms.mat')) and os.path.isfile(os.path.join(session_path,'behav.mat')):
                sessionList.append(session_path)#TODO add other conditions, like num cells, neurons, etc
            else:
                break

#%% Save list of sessions and stats in yaml files
sessions_dict = {'sessions':sessionList}
with open('batchList.yaml','w') as file:
    yaml.dump(sessions_dict,file)
                
dataset_stats_dict = {'numRegions': numRegions, 
                      'numSubjects': numSubjects, 
                      'numSessions': numSessions}

with open('output/dataset_stats.yaml','w') as file:
    yaml.dump(dataset_stats_dict,file)

# %%
session