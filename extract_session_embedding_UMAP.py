#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

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

#%% TODO create experimental folder, to save results, figs, params

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
#%% Plot embeddings
plt.figure(figsize=(3,1.5))
plt.subplot(121)
plt.scatter(train_embedding[:, 0], train_embedding[:, 1], s= 1, 
c=data['position'][data['trainingFrames'],0], cmap='Spectral')
plt.axis('equal')
plt.xlabel('$D_{1}$')
plt.ylabel('$D_{2}$')
plt.title('Train set')
plt.subplot(122)
plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s= 1, 
c=data['position'][~data['trainingFrames'],0], cmap='Spectral')
plt.axis('equal')
plt.xlabel('$D_{1}$')
plt.title('Test set')
plt.colorbar(label='Relative position')
plt.tight_layout()

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

# %% Reconstruct original data
full_embedding = embedding_model.transform(data['neuralData'][:,0:params['input_neurons']])
full_reconstruction = embedding_model.inverse_transform(full_embedding)
pred_position = pos_decoder.predict(full_embedding)
pred_time = time_decoder.predict(full_embedding)

#%% Plot summary
# Plot embedding results
plt.figure(figsize=(1,1))
cells2plot = 10
plt.subplot(121)
for i in range(cells2plot):
    plt.plot(data['caTime'],data['neuralData'][:,i]*params['plot_gain']+i,
#            c=(0,0,0),
            linewidth=.3)
plt.xlim(50,60)
plt.axis('off')

plt.subplot(122)
for i in range(cells2plot):
    plt.plot(data['caTime'],full_reconstruction[:,i]*params['plot_gain']+i,
#            c=(.8,0,0),
            linewidth=.3)
plt.xlim(50,60)
plt.axis('off')

# %%
#plt.figure(figsize=(1.5,1))
plt.scatter(full_embedding[:,0],full_embedding[:,1],c=data['position'][:,0], cmap='Spectral', s=1)
plt.title('embedding')
#plt.xlabel('$D_{1}$')
#plt.ylabel('$D_{2}$')
plt.axis('scaled')
plt.axis('off')
plt.colorbar(label='Location (cm)', fraction=0.025, pad=.001)

#%%
plt.figure(figsize=(.75,.75))
plt.title('reconstruction')
plt.scatter(data['neuralData'][~data['trainingFrames'],0:params['input_neurons']].flatten(),test_reconstruction.flatten(), s=1)
#plt.plot([0,100],[0,100],'r--')
#plt.xlim([0,100])
#plt.ylim([0,100])
plt.title(f'$R^2=${test_stats[0].round(4)}')
plt.xlabel('original')
plt.ylabel('reconstructed')

#%%
plt.figure(figsize=(.75,.75))
plt.title('location')
plt.scatter(data['position'].flatten(),pred_position.flatten(), s=1)
#plt.plot([0,100],[0,100],'r--')
#plt.xlim([0,100])
#plt.ylim([0,100])
plt.title(f'$R^2=${test_pred_pos_stats[0].round(4)}')
plt.xlabel('actual')
plt.ylabel('predicted')

#%%
plt.figure(figsize=(3,1))
plt.plot(data['caTime'],data['position'][:,0], label='Actual')
plt.plot([]);plt.plot([]);plt.plot([]);plt.plot([]);plt.plot([])
plt.plot(data['caTime'],pred_position[:,0], label='Decoded')
#plt.xlim([50,60])
#plt.ylim([0,100])
plt.xlabel('Time (s)')
plt.ylabel('Location (cm)')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
# %%
