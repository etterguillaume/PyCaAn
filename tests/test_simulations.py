#%%
from pycaan.functions.simulate import fit_ANNs
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells
ratinabox.autosave_plots = False
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import os

#%%
with open('../params_regions.yaml','r') as file:
    params = yaml.full_load(file)
# %%
path = '../../../datasets/calcium_imaging/CA1/M111/M111_smallOF_20190827'

#%%
data=load_data(path)
data = preprocess_data(data, params)
# %%
maze_width = {'OF':45,
                  'legoOF': 50,
                  'plexiOF': 49,
                  'smallOF': 38,
                  'LT': 100,
                  'legoLT':134,
                  'legoToneLT':134,
                  'legoSeqLT':134,
                  }

if data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF' or data['task']=='smallOF':
    environment = Environment(params={
    "scale": 1,
    'boundary':[[0,0],
                [0,maze_width[data['task']]/100],
                [maze_width[data['task']]/100,maze_width[data['task']]/100],
                [maze_width[data['task']]/100,0]]
    })
elif data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT': # For linear tracks
    environment = Environment(params={
    "scale": 1,
    'boundary':[[0,0],
                [0,0.1],
                [maze_width[data['task']]/100,0.1],
                [maze_width[data['task']]/100,0]]
    })

agent = Agent(environment)
agent.import_trajectory(times=data['caTime'], positions=data['position']/100) # Import existing coordinates

simulated_place_cells = PlaceCells(
    agent,
    params={
            "n": params['num_simulated_neurons'],
            "widths": params['sim_PC_widths'],
            })

simulated_grid_cells = GridCells(
    agent,
    params={
            "n": params['num_simulated_neurons'],
            "gridscale_distribution":'rayleigh',
            "gridscale": (.1,.5)
            })

dt = 1/params['sampling_frequency'] #TODO implement variable sampling rate
for i, t in enumerate(data['caTime']):
    agent.update(dt=dt)
    simulated_place_cells.update()
    simulated_grid_cells.update()

modeled_place_activity = np.array(simulated_place_cells.history['firingrate'])
modeled_grid_activity = np.array(simulated_grid_cells.history['firingrate'])

agent.plot_trajectory()



# %%
trainingFrames = np.zeros(len(data['caTime']), dtype=bool)

if params['train_set_selection']=='random':
    trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
elif params['train_set_selection']=='split':
    trainingFrames[0:int(params['train_test_ratio']*len(data['caTime']))] = True

testingFrames = ~trainingFrames

trainingFrames[~data['running_ts']] = False
testingFrames[~data['running_ts']] = False

modeled_place_activity = np.array(simulated_place_cells.history['firingrate'])
modeled_grid_activity = np.array(simulated_grid_cells.history['firingrate'])
# %%
from sklearn.preprocessing import StandardScaler
standardize = StandardScaler()
from sklearn.metrics import f1_score

#%%
num_neurons_used = 1024
port_gridcells_used = .75

scores = np.zeros(data['binaryData'].shape[1])*np.nan

# Sort neurons from best to worst for a given variable
for neuron_i in tqdm(range(data['binaryData'].shape[1])):
    num_GCs = int(port_gridcells_used*num_neurons_used)
    num_PCs = int((1-port_gridcells_used)*num_neurons_used)
    selected_PCs=np.random.choice(num_neurons_used,num_PCs)
    selected_GCs=np.random.choice(num_neurons_used,num_GCs)
    simulated_activity = standardize.fit_transform(np.concatenate((
        modeled_place_activity[:,selected_PCs],
        modeled_grid_activity[:,selected_GCs],
    ), axis=1))
    
    input_data = data['binaryData'][:,neuron_i].reshape(-1,1)
    model_neuron = LogisticRegression(
        penalty='l2',
        class_weight='balanced',
        random_state=params['seed']
    ).fit(simulated_activity[trainingFrames],
            input_data[trainingFrames])

    pred = model_neuron.predict(simulated_activity[testingFrames])

    scores[neuron_i]=f1_score(input_data[testingFrames],
                    pred)

#%%
plt.plot(scores)

# %%
