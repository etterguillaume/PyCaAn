#%%
from pycaan.functions.simulate import fit_ANNs
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons, PlaceCells, RandomSpatialNeurons, GridCells
from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons
ratinabox.autosave_plots = False
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import yaml
import os

#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)
# %%
path = '../../../datasets/calcium_imaging/CA1/M246/M246_OF_1'

#%%
data=load_data(path)
data = preprocess_data(data, params)
# %%
maze_width = {'OF':45,
                  'legoOF': 50,
                  'plexiOF': 49,
                  'LT': 100,
                  'legoLT':134,
                  'legoToneLT':134,
                  'legoSeqLT':134,
                  }

if data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
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
# %%
agent = Agent(environment)
agent.import_trajectory(times=data['caTime'], positions=data['position']/100) # Import existing coordinates

simulated_place_cells = PlaceCells(
    agent,
    params={
            "n": 256,
            "widths": .1,
            })
simulated_grid_cells = GridCells(
    agent,
    params={
            "n": 256,
            "gridscale": (.1,.5),
            })
# %% Simulate
test=[]
previous_t = 0
for i, t in enumerate(data['caTime']):
    dt = 1/params['sampling_frequency']
    test.append(dt)
    agent.update(dt=dt)
    simulated_place_cells.update()
    simulated_grid_cells.update()
    previous_t=t
# %%
trainingFrames = np.zeros(len(data['caTime']), dtype=bool)

if params['train_set_selection']=='random':
    trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
elif params['train_set_selection']=='split':
    trainingFrames[0:int(params['train_test_ratio']*len(data['caTime']))] = True

testingFrames = ~trainingFrames

trainingFrames[~data['running_ts']] = False
testingFrames[~data['running_ts']] = False

place_cells_activity = np.array(simulated_place_cells.history['firingrate'])
grid_cells_activity = np.array(simulated_grid_cells.history['firingrate'])
# %%
from sklearn.preprocessing import StandardScaler
standardize = StandardScaler()
from sklearn.metrics import f1_score

neuron_ID=5

place_model_neuron = LogisticRegression(
                                        class_weight='balanced',
                                        penalty='l2',
                                        random_state=params['seed']).fit(standardize.fit_transform(place_cells_activity[trainingFrames]),
                                                                        data['binaryData'][trainingFrames,neuron_ID])
grid_model_neuron = LogisticRegression(
                                       class_weight='balanced',
                                       penalty='l2',
                                       random_state=params['seed']).fit(standardize.fit_transform(grid_cells_activity[trainingFrames]),
                                                                        data['binaryData'][trainingFrames,neuron_ID])

PC_score=place_model_neuron.score(standardize.fit_transform(place_cells_activity[testingFrames]),
                                                                        data['binaryData'][testingFrames,neuron_ID])
PC_pred = place_model_neuron.predict(standardize.fit_transform(place_cells_activity[testingFrames]))
PC_Fscore=f1_score(data['binaryData'][testingFrames,neuron_ID], PC_pred)

GC_score=grid_model_neuron.score(standardize.fit_transform(grid_cells_activity[testingFrames]),
                                                                        data['binaryData'][testingFrames,neuron_ID])
GC_pred = grid_model_neuron.predict(standardize.fit_transform(grid_cells_activity[testingFrames]))
GC_Fscore=f1_score(data['binaryData'][testingFrames,neuron_ID], GC_pred)


# %%
PC_model_prediction_DF = place_model_neuron.decision_function(place_cells_activity[testingFrames])
GC_model_prediction_DF = grid_model_neuron.decision_function(grid_cells_activity[testingFrames])
# %%
