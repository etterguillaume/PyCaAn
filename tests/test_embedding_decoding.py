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

#%%
data = extract_seqLT_tone(data,params)

#%%
decoding_score, z_score, p_value, _, _, _ = decode_embedding(data['seqLT_state'],data, params, train_embedding, test_embedding)

#%%





#%%
#with h5py.File(os.path.join(working_directory,'direction_decoding.h5'),'w') as f:
if data['task'] == 'OF' or data['task'] == 'legoOF' or data['task'] == 'plexiOF':
    decoding_score, z_score, p_value, decoding_error, shuffled_error, test_prediction = decode_embedding(data['heading'],data, params, train_embedding, test_embedding)
    #   f.create_dataset('decoding_error', data=decoding_error)
#    f.create_dataset('shuffled_error', data=shuffled_error)
elif data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
    decoding_score, z_score, p_value, _, _, _ = decode_embedding(data['LT_direction'], data, params, train_embedding, test_embedding)

 #   f.create_dataset('decoding_score', data=decoding_score)
  #  f.create_dataset('z_score', data=z_score)
   # f.create_dataset('p_value', data=p_value)
# %%
