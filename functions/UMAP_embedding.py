#%% Import dependencies
import yaml
import numpy as np
import matplotlib.pyplot as plt
#from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data

#%% Load parameters
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% TODO parameter override here

#%% Load session
session_path = '../../datasets/calcium_imaging/M246/M246_LT_6'
data = load_data(session_path)

#%% Preprocessing 
data = preprocess_data(data,params)

#%% Split dataset into
np.random.seed(params['seed'])
trainingFrames = np.zeros(len(data['caTime']), dtype=bool)

if params['train_set_selection']=='random':
    trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
elif params['train_set_selection']=='split':
    trainingFrames[0:int(params['train_test_ratio']*len(data['caTime']))] = True 
data['trainingFrames']=trainingFrames

#%% Train embedding model
embedding_model = ParametricUMAP(
                       parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                       autoencoder_loss = False,
                       parametric_reconstruction= True,
                       n_components=params['embedding_dims'],
                       n_neighbors=params['n_neighbors'],
                       min_dist=params['min_dist'],
                       metric='euclidean',
                       random_state=42).fit(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])

#%%
train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
test_embedding = embedding_model.transform(data['neuralData'][~data['trainingFrames'],0:params['input_neurons']])

#%% Reconstruct inputs for both train and test sets
test_reconstruction = embedding_model.inverse_transform(test_embedding)
test_stats = corr(data['neuralData'][~data['trainingFrames'],0:params['input_neurons']].flatten(),test_reconstruction.flatten())

#%% Train decoder
pos_decoder = LinearRegression().fit(train_embedding, data['position'][data['trainingFrames'],:])
train_pred_pos = pos_decoder.predict(train_embedding)
test_pred_pos = pos_decoder.predict(test_embedding)
#score=pos_decoder.score(test_embedding, data['position'][~data['trainingFrames'],:])
time_decoder = LinearRegression().fit(train_embedding, data['caTime'][data['trainingFrames']])
train_pred_time = time_decoder.predict(train_embedding)
test_pred_time = time_decoder.predict(test_embedding)

#%% Decoding accuracy
train_pred_pos_stats = corr(train_pred_pos.flatten(),data['position'][data['trainingFrames'],:].flatten())
test_pred_pos_stats = corr(test_pred_pos.flatten(),data['position'][~data['trainingFrames'],:].flatten())
train_pred_time_stats = corr(train_pred_time.flatten(),data['caTime'][data['trainingFrames']].flatten())
test_pred_time_stats = corr(test_pred_time.flatten(),data['caTime'][~data['trainingFrames']].flatten())

#%% Decoding error

#%% Save results
# Save experiment name (mouse, session, maze, region?)
# Save params
# Save error, accuracy, etc