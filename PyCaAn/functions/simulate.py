#%% Imports
import os
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons, PlaceCells, RandomSpatialNeurons, GridCells
ratinabox.autosave_plots = False
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
standardize = StandardScaler()

def fit_ANNs(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])

    # Create folder with convention (e.g. CA1_M246_LT_2017073)
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

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

    agent = Agent(environment)
    agent.import_trajectory(times=data['caTime'], positions=data['position']/100) # Import existing coordinates

    simulated_place_cells = PlaceCells(
        agent,
        params={
                "n": params['num_artificial_PCs'],
                "widths": .1,
                })
    simulated_grid_cells = GridCells(
        agent,
        params={
                "n": params['num_artificial_GCs'],
                "gridscale": (.1,.5),
                })

    dt = 1/params['sampling_frequency'] #TODO implement variable sampling rate
    for i, t in enumerate(data['caTime']):
        agent.update(dt=dt)
        simulated_place_cells.update()
        simulated_grid_cells.update()

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

    PC_scores = np.zeros(data['binaryData'].shape[1])*np.nan
    PC_Fscores = np.zeros(data['binaryData'].shape[1])*np.nan
    GC_scores = np.zeros(data['binaryData'].shape[1])*np.nan
    GC_Fscores = np.zeros(data['binaryData'].shape[1])*np.nan

    for neuron_i in range(data['binaryData'].shape[1]):
        if sum(data['binaryData'][trainingFrames,neuron_i])>0:
            place_model_neuron = LogisticRegression(
                                            class_weight='balanced',
                                            penalty='l2',
                                            random_state=params['seed']).fit(standardize.fit_transform(place_cells_activity[trainingFrames]),
                                                                            data['binaryData'][trainingFrames,neuron_i])
            grid_model_neuron = LogisticRegression(
                                                class_weight='balanced',
                                                penalty='l2',
                                                random_state=params['seed']).fit(standardize.fit_transform(grid_cells_activity[trainingFrames]),
                                                                                    data['binaryData'][trainingFrames,neuron_i])

            PC_scores[neuron_i]=place_model_neuron.score(standardize.fit_transform(place_cells_activity[testingFrames]),
                                                                                    data['binaryData'][testingFrames,neuron_i])
            PC_pred = place_model_neuron.predict(standardize.fit_transform(place_cells_activity[testingFrames]))
            PC_Fscores[neuron_i] = f1_score(data['binaryData'][testingFrames,neuron_i], PC_pred)

            GC_scores[neuron_i] = grid_model_neuron.score(standardize.fit_transform(grid_cells_activity[testingFrames]),
                                                                                    data['binaryData'][testingFrames,neuron_i])
            GC_pred = grid_model_neuron.predict(standardize.fit_transform(grid_cells_activity[testingFrames]))
            GC_Fscores[neuron_i] = f1_score(data['binaryData'][testingFrames,neuron_i], GC_pred)
        
    return PC_scores, PC_Fscores, GC_scores, GC_Fscores

def simulate_activity(recording_length, num_bins, ground_truth_info, sampling):
    # Use this function to simulate binarized calcium activity
    assert num_bins>1
    if num_bins==2:
        variable = np.ones(recording_length)
    else:
        variable = np.random.choice(np.arange(1,num_bins),recording_length) # randomly sample bins
    activity = np.zeros(recording_length,dtype=bool)

    variable[np.random.choice(np.arange(recording_length), int(sampling*recording_length), replace=False)] = 0 # Set number of samples for which activity could predict the variable
    activity[variable==0]=True # Neural activity perfectly predicts variable

    bits2flip = np.random.choice(np.arange(recording_length), int(recording_length*(1-ground_truth_info)), replace=False)
    bits2flip = np.random.choice(bits2flip,int(0.5*len(bits2flip))) # Flip a coin to decide whether to flip those bits
    activity[bits2flip] = ~activity[bits2flip]

    return activity, variable