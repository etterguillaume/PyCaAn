#%% Imports
import os
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells, BoundaryVectorCells
ratinabox.autosave_plots = False
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

@ignore_warnings(category=ConvergenceWarning)
def fit_ANNs(data, params, modeled_place_activity, modeled_grid_activity, modeled_BVC_activity):
    trainingFrames = np.zeros(len(data['caTime']), dtype=bool)

    if params['train_set_selection']=='random':
        trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
    elif params['train_set_selection']=='split':
        trainingFrames[0:int(params['train_test_ratio']*len(data['caTime']))] = True 

    testingFrames = ~trainingFrames

    trainingFrames[~data['running_ts']] = False
    testingFrames[~data['running_ts']] = False

    # Extract standardization on train set to avoid leaking dataset stats into test set
    place_standardizer = StandardScaler()
    grid_standardizer = StandardScaler()
    BVC_standardizer = StandardScaler()
    
    place_standardizer.fit(modeled_place_activity[trainingFrames])
    grid_standardizer.fit(modeled_grid_activity[trainingFrames])
    BVC_standardizer.fit(modeled_BVC_activity[trainingFrames])

    Fscores_placeModel = np.zeros(data['binaryData'].shape[1])*np.nan
    Fscores_gridModel = np.zeros(data['binaryData'].shape[1])*np.nan
    Fscores_BVCModel = np.zeros(data['binaryData'].shape[1])*np.nan

    # Sort neurons from best to worst for a given variable
    for neuron_i in range(data['binaryData'].shape[1]):
        if sum(data['binaryData'][trainingFrames, neuron_i])>0:
            place_model = LogisticRegression(verbose=False,
                                            class_weight='balanced',
                                            penalty='l2',
                                            random_state=params['seed']).fit(place_standardizer.transform(modeled_place_activity[trainingFrames]),
                                                                            data['binaryData'][trainingFrames,neuron_i])

            grid_model = LogisticRegression(
                                            class_weight='balanced',
                                            penalty='l2',
                                            random_state=params['seed']).fit(grid_standardizer.transform(modeled_grid_activity[trainingFrames]),
                                                                            data['binaryData'][trainingFrames,neuron_i])
            
            BVC_model = LogisticRegression(
                                            class_weight='balanced',
                                            penalty='l2',
                                            random_state=params['seed']).fit(BVC_standardizer.transform(modeled_grid_activity[trainingFrames]),
                                                                            data['binaryData'][trainingFrames,neuron_i])

            place_pred = place_model.predict(place_standardizer.transform(modeled_place_activity[testingFrames]))
            grid_pred = grid_model.predict(grid_standardizer.transform(modeled_grid_activity[testingFrames]))
            BVC_pred = BVC_model.predict(BVC_standardizer.transform(modeled_BVC_activity[testingFrames]))

            Fscores_placeModel[neuron_i] = f1_score(data['binaryData'][testingFrames,neuron_i], place_pred)
            Fscores_gridModel[neuron_i] = f1_score(data['binaryData'][testingFrames,neuron_i], grid_pred)
            Fscores_BVCModel[neuron_i] = f1_score(data['binaryData'][testingFrames,neuron_i], BVC_pred)
            
    return Fscores_placeModel, Fscores_gridModel, Fscores_BVCModel

def model_data(data, params):
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
    
    simulated_BVCs = BoundaryVectorCells(
        agent,
        params={
                "n": params['num_simulated_neurons'],
                "cell_arrangement": "random",
                "distance_distribution": "uniform",
                "distance_distribution_params": [.1,.5],
                })

    dt = 1/params['sampling_frequency'] #TODO implement variable sampling rate
    for i, t in enumerate(data['caTime']):
        agent.update(dt=dt)
        simulated_place_cells.update()
        simulated_grid_cells.update()
        simulated_BVCs.update()
    
    modeled_place_activity = np.array(simulated_place_cells.history['firingrate'])
    modeled_grid_activity = np.array(simulated_grid_cells.history['firingrate'])
    modeled_BVC_activity = np.array(simulated_BVCs.history['firingrate'])
        
    return modeled_place_activity, modeled_grid_activity, modeled_BVC_activity

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