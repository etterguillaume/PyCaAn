#%%
%load_ext autoreload
%autoreload 2

#%%
import yaml
import numpy as np
import joblib

from torch import nn
import torch
import torch.nn.functional as F

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
with open('../params_CA1.yaml','r') as file:
    params = yaml.full_load(file)

#%%
working_directory = '../../../output/results_CA1/CA1_M986_legoSeqLT_20190312'

#%%
session = '../../../datasets/calcium_imaging/CA1/M986/M986_legoSeqLT_20190312'
data = preprocess_data(load_data(session),params)

#%%
embedding_file = h5py.File(os.path.join(working_directory,'embedding.h5'),'r')
embedding = embedding_file['embedding'][()]
train_embedding = embedding_file['train_embedding'][()]
test_embedding = embedding_file['test_embedding'][()]
trainingFrames = embedding_file['trainingFrames'][()]
testingFrames = embedding_file['testingFrames'][()]
data['testingFrames'] = testingFrames
data['trainingFrames'] = trainingFrames
bin_vec=np.arange(100)

#%% Implement griddata
from scipy.interpolate import griddata
# %%
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.neighbors import KNeighborsRegressor as knn_reg

#%% Inverse decoding
# First, predict manifold from behavior in mouse A
ref_manifold_predictor = knn_reg(metric='euclidean', n_neighbors=15).fit(data['position'][data['trainingFrames'],:], train_embedding)

# Next, predict manifold from B given behavior from B and decoder from A
pred_target_manifold = ref_manifold_predictor.predict(data['position'][data['testingFrames'],:])
quantized_embedding = griddata(train_embedding, data['position'][trainingFrames,0], (data['position'][testingFrames,0]*np.ones((4,1))).T, method='nearest')
#manifold_aligner = lin_reg().fit(quantized_embedding,train_embedding)

#%%


decoder_var_ref = knn_reg(metric='euclidean', n_neighbors=15).fit(train_embedding, data['position'][data['trainingFrames'],0])

#%% Pipeline approach
from sklearn.pipeline import Pipeline
model = Pipeline(steps=[('decode_behavior', decoder_var_ref)]) # Include pre-trained decoder

# %% Train pipleline
model.fit(test_embedding, data['position'][data['testingFrames'],0])



#%%
data['position'][data['trainingFrames'],0]


decoder_var_pred = (lin_reg().fit(train_embedding,)
# %%
('align_embeddings', lin_reg()), 