#%%
%load_ext autoreload
%autoreload 2

#%%
import yaml
import numpy as np
import joblib
from umap.umap_ import UMAP
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import tensorflow as tf
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression as lin_reg
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.decoding import decode_embedding
from pycaan.functions.signal_processing import extract_tone, extract_seqLT_tone
import h5py
#%%
with open('../params.yaml','r') as file:
    params = yaml.full_load(file)

# #%% Load folders to analyze from yaml file?
# with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
#     session_file = yaml.full_load(file)
# session_list = session_file['sessions']
# path = session_list[232]
#%%
path = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_5'
data = load_data(path)
data = preprocess_data(data,params)
# %%
trainingFrames = np.zeros(len(data['caTime']), dtype=bool)

if params['train_set_selection']=='random':
    trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
elif params['train_set_selection']=='split':
    trainingFrames[0:int(params['train_test_ratio']*len(data['caTime']))] = True 

data['trainingFrames'] = trainingFrames
data['testingFrames'] = ~trainingFrames

# Exclude immobility from all sets
data['trainingFrames'][~data['running_ts']] = False
data['testingFrames'][~data['running_ts']] = False

#tf.keras.backend.clear_session()

# Train embedding model
embedding_model = UMAP(
                  #  verbose=False,
                   # parametric_reconstruction_loss_fcn=tf.keras.losses.MeanSquaredError(),
                   # autoencoder_loss = True,
                   # parametric_reconstruction = True,
                    n_components=params['embedding_dims'],
                    n_neighbors=params['n_neighbors'],
                    min_dist=params['min_dist'],
                    metric='euclidean',
                    random_state=42
                    ).fit(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])

#
#train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
train_embedding = embedding_model.transform(data['neuralData'][data['trainingFrames'],0:params['input_neurons']])
test_embedding = embedding_model.transform(data['neuralData'][data['testingFrames'],0:params['input_neurons']])
total_embedding = embedding_model.transform(data['neuralData'][:,0:params['input_neurons']])

# Reconstruct inputs for both train and test sets
reconstruction = embedding_model.inverse_transform(test_embedding)

# Assess reconstruction error
reconstruction_decoder = lin_reg().fit(reconstruction, data['rawData'][data['testingFrames']])
reconstruction_score = reconstruction_decoder.score(reconstruction, data['rawData'][data['testingFrames']])
# %%
embedding_model.save('model.h5')
#joblib.dump(embedding_model, 'model.pkl')

# %%
embedding_model2 = load_ParametricUMAP(os.path.join('model.h5'))
#embedding_model2 = joblib.load("model.pkl")
# %%
