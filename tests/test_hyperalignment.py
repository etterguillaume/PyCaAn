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
working_directory_A = '../../../output/results_CA1/CA1_M246_LT_4'
working_directory_B = '../../../output/results_CA1/CA1_M246_LT_6'

#%%
session_A = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_4'
session_B = '../../../datasets/calcium_imaging/CA1/M246/M246_LT_6'
data_A = preprocess_data(load_data(session_A),params)
data_B = preprocess_data(load_data(session_B),params)

#%%
embedding_file_A = h5py.File(os.path.join(working_directory_A,'embedding.h5'),'r')
embedding_file_B = h5py.File(os.path.join(working_directory_B,'embedding.h5'),'r')
embedding_A = embedding_file_A['embedding'][()]
embedding_B = embedding_file_B['embedding'][()]
train_embedding_A = embedding_file_A['train_embedding'][()]
train_embedding_B = embedding_file_B['train_embedding'][()]
test_embedding_A = embedding_file_A['test_embedding'][()]
test_embedding_B = embedding_file_B['test_embedding'][()]
trainingFrames_A = embedding_file_A['trainingFrames'][()]
trainingFrames_B = embedding_file_B['trainingFrames'][()]
testingFrames_A = embedding_file_A['testingFrames'][()]
testingFrames_B = embedding_file_B['testingFrames'][()]
bin_vec=(np.linspace(0,100,params['quantization_steps']))

#%%
from pycaan.functions.embedding import quantize_embedding, extract_hyperalignment_score

#%%
from scipy.interpolate import interp1d
def quantize_embedding(embedding, var, bin_vec):
        # Digitize labels
        quantized_var=np.digitize(var, bin_vec, right=False)

        # Quantize and align embeddings
        quantized_embedding = np.zeros((len(bin_vec),embedding.shape[1]))

        for i, bin_val in enumerate(bin_vec):
                quantized_embedding[i,:] = np.nanmean(embedding[quantized_var==i],axis=0)

        # Interpolate through missing values
        for dim in range(quantized_embedding.shape[1]):
                nan_vec = np.isnan(quantized_embedding[:,dim])
                interp_func = interp1d(bin_vec[~nan_vec],
                                        quantized_embedding[~nan_vec,dim],
                                        fill_value="extrapolate")
                quantized_embedding[:,dim] = interp_func(quantized_embedding[:,dim])

        return quantized_embedding
#%%
train_quantized_embedding_B = quantize_embedding(train_embedding_B,
                                                        data_B['position'][trainingFrames_B,0], 
                                                        bin_vec)
test_quantized_embedding_B = quantize_embedding(test_embedding_B,
                                                    data_B['position'][testingFrames_B,0], 
                                                    bin_vec)

#%%
train_quantized_embedding_B

#%%
import matplotlib.pyplot as plt
plt.scatter(train_quantized_embedding_A[:,3], train_quantized_embedding_B[:,3], c=bin_vec)

#%%
extract_hyperalignment_score(embedding_A,
                                data_A['position'][:,0],
                                trainingFrames_A,
                                testingFrames_A,
                                embedding_B,
                                data_B['position'][:,0],
                                trainingFrames_B,
                                testingFrames_B,
                                bin_vec)


#%%












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