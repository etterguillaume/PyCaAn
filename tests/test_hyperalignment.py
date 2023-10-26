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

# %%
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.neighbors import KNeighborsRegressor as knn_reg
#%%
decoder_var_ref = knn_reg(metric='euclidean', n_neighbors=15).fit(train_embedding, data['position'][data['trainingFrames'],0])
# %%
decoder_var_pred = decoder_var_ref.predict(lin_reg().fit(train_embedding, data['position'][data['trainingFrames'],0]))
# %%
