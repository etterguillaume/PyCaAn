#%% TEMP
%load_ext autoreload
%autoreload 2

#%% Imports
import yaml
import os
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.tuning import extract_1D_tuning, extract_2D_tuning
from functions.metrics import extract_total_distance_travelled
import matplotlib.pyplot as plt
import h5py

#%% Load YAML file
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% List all sessions here, tqdm
session_list=['../../datasets/calcium_imaging/CA1/M246/M246_legoOF_20180621']

#%% Load data
data = load_data(session_list[0])

#%% Check if data already analyzed
working_directory = os.path.join('output',data['region'],data['subject'],str(data['day']))
if not os.path.exists(working_directory): # If folder does not exist, create it
    os.mkdir(working_directory)

#%% Pre-process data
data=preprocess_data(data,params)

#%% Save basic info
numFrames, numNeurons = data['rawData'].shape
total_distance_travelled = extract_total_distance_travelled(data['position'])

info_dict = {
            'day': data['day'],
            'task': data['task'],
            'subject': data['subject'],
            'region': data['region'],
            'sex': data['sex'],
            'age': data['age'],
            'condition': data['day'],
            'darkness': data['darkness'],
            'optoStim': data['optoStim'],
            'rewards': data['rewards'],
            'darkness': data['darkness'],
            'condition': data['condition'],
            'numNeurons': numNeurons,
            'numFrames': numFrames,
            'total_distance_travelled': total_distance_travelled,
            'duration': data['caTime'][-1]
    }

with h5py.File(os.path.join(working_directory,'info.hdf5'), "a") as f:
    for k, v in info_dict.items():
        f.create_dataset(k, data=v)
    #dset = f.create_group("info", data=info_dict)
#TODO extract basic stats, num cells, distance travelled, etc in info.h5

#%%
f = h5py.File(os.path.join(working_directory,'info.hdf5'), "r")


#%% Extract tuning curves
# Always check if data exists before analyzing!!
# save tuning_curves and in individual files (e.g location.h5, velocity.h5,... which contains tuning_curve, AMI, occupancy, etc for each neuron)

# For all:
# extract elapsed time tuning
# extract distance travelled tuning
# extract velocity tuning

# for all, maze-size dependent:
# extract location

# for all, maze-type dependent (LT, non-LT)
# extract LR direction tuning OR extract HD tuning

# for toneLT, seqLT
# extract tone tuning

# for seqLT
# extract tone type tuning


# TODO check if folder exists, check if params['overwrite'] is 'changes_only', 'always' or 'never'
# TODO if exists, check results contents. If params_old==params_new, decide whether overwrite

#%% Extract tuning correlatins

#%% Decode (OPTIONAL)




