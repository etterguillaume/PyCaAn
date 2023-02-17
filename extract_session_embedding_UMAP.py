#%%TEMP FOR DEBUG
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Import dependencies
import yaml
import numpy as np
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
from sklearn.svm import SVC
from scipy.stats import pearsonr as corr
from functions.dataloaders import load_data
from functions.signal_processing import preprocess_data
from functions.analysis import analyze_binary_reconstruction, analyze_decoding
from functions.plotting import plot_losses, plot_embedding_results_raw, plot_embedding_results_binary

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
np.random.seed(42)
trainingFrames = np.zeros(len(data['caTime']), dtype=bool)
trainingFrames[np.random.choice(np.arange(len(data['caTime'])), size=int(len(data['caTime'])*params['train_test_ratio']), replace=False)] = True
data['trainingFrames']=trainingFrames

#%% Train embedding model
embedding_model = UMAP(n_neighbors=50,
                       min_dist=.1,
                       n_components=params['embedding_dims'],
                       metric='euclidean',
                       random_state=42).fit(data['procData'][data['trainingFrames'],0:params['input_neurons']])

#%%
train_embedding = embedding_model.transform(data['procData'][data['trainingFrames'],0:params['input_neurons']])
test_embedding = embedding_model.transform(data['procData'][~data['trainingFrames'],0:params['input_neurons']])
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
train_reconstruction = embedding_model.inverse_transform(train_embedding)
test_reconstruction = embedding_model.inverse_transform(test_embedding)

#%%
train_stats = corr(data['procData'][data['trainingFrames'],0:params['input_neurons']].flatten(),train_reconstruction.flatten())
test_stats = corr(data['procData'][~data['trainingFrames'],0:params['input_neurons']].flatten(),test_reconstruction.flatten())

# %% Compute reconstruction accuracy if binarized
if params['data_type']=='binarized':
    train_accuracy, train_precision, train_recall, train_F1 = analyze_binary_reconstruction(params, embedding_model, train_loader)
    test_accuracy, test_precision, test_recall, test_F1 = analyze_binary_reconstruction(params, embedding_model, test_loader)
    print(f'Train F1: {np.mean(train_F1).round(4)}, Test F1: {np.mean(test_F1).round(4)}')

#%% Train decoder
pos_decoder = LinearRegression().fit(train_embedding, data['position'][data['trainingFrames'],:])
score=pos_decoder.score(test_embedding, data['position'][~data['trainingFrames'],:])
test_pred_pos = pos_decoder.predict(test_embedding)
#%%

test_pred_pos_stats = corr(test_pred_pos.flatten(),data['position'][~data['trainingFrames'],:].flatten())




# %% Compute decoding errors #TODO only when speed > threshold
train_decoding_error, train_decoder_stats = analyze_decoding(params, embedding_model, embedding_decoder, train_loader)
test_decoding_error, test_decoder_stats = analyze_decoding(params, embedding_model, embedding_decoder, test_loader)





# %% Reconstruct original data
original = torch.tensor(data['procData'][:,0:params['input_neurons']],dtype=torch.float)
reconstruction, embedding = embedding_model(original)
pred_position = embedding_decoder(embedding)

#%%
if params['data_type']=='raw':
    reconstruction_R, p_value = corr(original.flatten(),reconstruction.detach().flatten())
    plot_embedding_results_raw(params, original, reconstruction.detach(), embedding, reconstruction_R, test_decoder_stats[0], data['position'][:,0], pred_position[:,0].detach(), data['velocity'])
elif params['data_type']=='binarized':
    plot_embedding_results_binary(original, reconstruction, embedding, test_F1, test_decoder_stats[0], data['position'][:,0], pred_position[:,0].detach(), data['velocity'])

# %% Save results
