#%%
import numpy as np
import os
import yaml
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers.core import Lambda
from umap.umap_ import UMAP
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.neighbors import KNeighborsRegressor as KNN_reg
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use('plot_style.mplstyle')

#%% Import params
with open('params.yaml','r') as file:
    params = yaml.full_load(file)

#%% Snippet from piVAE paper (Zhou & Wei, 2020)
def perm_func(x, ind):
    """Utility function. Permute x with given indices. Search tf.gather for detailed use of this function.
    """
    return tf.gather(x, indices=ind, axis=-1);

def slice_func(x, start, size):
    """Utility function. We use it to take a slice of tensor start from 'start' with length 'size'. Search tf.slice for detailed use of this function.
    
    """
    return tf.slice(x, [0,start],[-1,size])

def realnvp_layer(x_input):
    DD = x_input.shape.as_list()[-1]; ## DD needs to be an even number
    dd = (DD//2);
    
    ## define some lambda functions
    clamp_func = Lambda(lambda x: 0.1*tf.tanh(x));
    trans_func = Lambda(lambda x: x[0]*tf.exp(x[1]) + x[2]);
    sum_func = Lambda(lambda x: K.sum(-x, axis=-1, keepdims=True));
    
    ## compute output for s and t functions
    x_input1 = Lambda(slice_func, arguments={'start':0,'size':dd})(x_input);
    x_input2 = Lambda(slice_func, arguments={'start':dd,'size':dd})(x_input);
    st_output = x_input1;
    
    n_nodes = [dd//2, dd//2, DD];
    act_func = ['relu', 'relu', 'linear'];
    for ii in range(len(act_func)):
        st_output = layers.Dense(n_nodes[ii], activation = act_func[ii])(st_output);
    s_output = Lambda(slice_func, arguments={'start':0,'size':dd})(st_output);
    t_output = Lambda(slice_func, arguments={'start':dd,'size':dd})(st_output);
    s_output = clamp_func(s_output); ## keep small values of s
    
    ## perform transformation
    trans_x = trans_func([x_input2, s_output, t_output]);
    output = layers.concatenate([trans_x, x_input1], axis=-1);
    return output

def realnvp_block(x_output):
    for _ in range(2):
        x_output = realnvp_layer(x_output);
    return x_output

def simulate_cont_data(length, n_dim):
    ## simulate 2d z
    np.random.seed(777);
    
    u_true = np.random.uniform(0,2*np.pi,size = [length,1]);
    mu_true = np.hstack((u_true, 2*np.sin(u_true)));
    z_true = np.random.normal(0, 0.6, size=[length,2])+mu_true;
    z_true = np.hstack((z_true, np.zeros((z_true.shape[0],n_dim-2))));
    
    ## simulate mean
    dim_x = z_true.shape[-1];
    permute_ind = [];
    n_blk = 4;
    for ii in range(n_blk):
        np.random.seed(ii);
        permute_ind.append(tf.convert_to_tensor(np.random.permutation(dim_x)));
    
    x_input = layers.Input(shape=(dim_x,));
    x_output = realnvp_block(x_input);
    for ii in range(n_blk-1):
        x_output = Lambda(perm_func, arguments={'ind':permute_ind[ii]})(x_output);
        x_output = realnvp_block(x_output);
    
    realnvp_model = Model(inputs=[x_input], outputs=x_output);
    mean_true = realnvp_model.predict(z_true)
    lam_true = np.exp(2.2*np.tanh(mean_true));
    return z_true, u_true, mean_true, lam_true

#%% Generate artificial data
length = 10000
train_idx = np.arange(0,int(params['train_test_ratio']*length))
n_dim = 128
z_true, u_true, mean_true, lam_true = simulate_cont_data(length, n_dim)

#%%
embedding_model = UMAP(
                       n_components=params['embedding_dims'],
                       n_neighbors=params['n_neighbors'],
                       min_dist=params['min_dist'],
                       metric='euclidean',
                       random_state=42).fit(mean_true[train_idx])

#%% Transform data
train_embedding = embedding_model.transform(mean_true[train_idx])
test_embedding = embedding_model.transform(mean_true[~train_idx])

#%% Test embedding performance
embedding_decoder=lin_reg().fit(train_embedding, z_true[train_idx,0:2])
embedding_perf = embedding_decoder.score(test_embedding,z_true[~train_idx,0:2])

#%% Reconstruct inputs
train_reconstruction = embedding_model.inverse_transform(train_embedding)
test_reconstruction = embedding_model.inverse_transform(test_embedding)

#%% Assess reconstruction performance
reconstruction_decoder=lin_reg().fit(train_reconstruction, mean_true[train_idx])
reconstruction_perf = reconstruction_decoder.score(test_reconstruction,mean_true[~train_idx])

# %%
plt.figure()
plt.title('True latent')
plt.scatter(z_true[~train_idx,0], z_true[~train_idx,1], edgecolors='none', s=.5, c=u_true[~train_idx], cmap='Spectral')
plt.axis('equal')
plt.axis('off')
plt.colorbar(label='Artificial activity')
plt.savefig(os.path.join(params['path_to_results'],'figures','ground_truth_manifold.pdf'))

# %%
plt.figure()
plt.title(f'Embedding performance\nR$^2$: {embedding_perf.round(4)}')
plt.scatter(test_embedding[:,0], test_embedding[:,1], edgecolors='none', s=.5, c=u_true[~train_idx], cmap='Spectral')
plt.axis('equal')
plt.axis('off')
plt.colorbar(label='Artificial activity')
plt.savefig(os.path.join(params['path_to_results'],'figures','estimated_manifold.pdf'))

# %%
plt.figure()
plt.title(f'Reconstruction accuracy\nR$^2${reconstruction_perf.round(4)}')
plt.scatter(mean_true[~train_idx].flatten()[::100], test_reconstruction.flatten()[::100], edgecolors='none', s=.5)
plt.axis('equal')
plt.xlabel('Original')
plt.ylabel('Reconstruction')
plt.savefig(os.path.join(params['path_to_results'],'figures','GT_vs_reconstruction_performance.pdf'))

# %%
decoder = KNN_reg(metric='euclidean').fit(train_embedding, u_true[train_idx])
prediction = decoder.predict(test_embedding)

prediction_stats = decoder.score(test_embedding,u_true[~train_idx])
# %%
plt.figure()
plt.plot(u_true[~train_idx],label='Actual')
plt.plot([]);plt.plot([]);plt.plot([]);plt.plot([]);plt.plot([]) # Skip cycler colors
plt.plot(prediction,label='Decoded')
plt.xlim([0,50])
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend(bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0)
plt.savefig(os.path.join(params['path_to_results'],'figures','simulated_decoded_manifold.pdf'))

#%%
plt.title(f'Decoding accuracy\nR$^2${prediction_stats.round(4)}')
plt.scatter(u_true[~train_idx].flatten(), prediction.flatten(), edgecolors='none', s=.5)
plt.plot([0,6],[0,6],'C6--')
plt.axis('equal')
plt.xlabel('Original')
plt.ylabel('Reconstruction')
plt.savefig(os.path.join(params['path_to_results'],'figures','simulated_decoding_performance.pdf'))

# %% Optimize parameters
num_neurons_list=[8,16,32,64,128,256,512,1024]

embedding_perf = np.zeros(len(num_neurons_list))*np.nan
reconstruction_perf = np.zeros(len(num_neurons_list))*np.nan
decoder_perf = np.zeros(len(num_neurons_list))*np.nan

for i, num_neurons in enumerate(tqdm(num_neurons_list)):
    z_true, u_true, mean_true, lam_true = simulate_cont_data(length, num_neurons)

    embedding_model = UMAP(
                       n_components=params['embedding_dims'],
                       n_neighbors=params['n_neighbors'],
                       min_dist=params['min_dist'],
                       metric='euclidean',
                       random_state=42).fit(mean_true[train_idx])

    train_embedding = embedding_model.transform(mean_true[train_idx])
    test_embedding = embedding_model.transform(mean_true[~train_idx])

    embedding_decoder=lin_reg().fit(train_embedding, z_true[train_idx,0:2])
    embedding_perf[i] = embedding_decoder.score(test_embedding,z_true[~train_idx,0:2])

    train_reconstruction = embedding_model.inverse_transform(train_embedding)
    test_reconstruction = embedding_model.inverse_transform(test_embedding)

    reconstruction_decoder=lin_reg().fit(train_reconstruction, mean_true[train_idx])
    reconstruction_perf[i] = reconstruction_decoder.score(test_reconstruction,mean_true[~train_idx])

    decoder = KNN_reg(metric='euclidean').fit(train_embedding, u_true[train_idx])
    prediction = decoder.predict(test_embedding)

    decoder_perf[i] = decoder.score(test_embedding,u_true[~train_idx])

#%% Plot results
plt.figure()
plt.plot(num_neurons_list,embedding_perf)
plt.xlabel('Neurons')
plt.ylabel('Embedding performance')
plt.savefig(os.path.join(params['path_to_results'],'figures','optim_numNeurons_embedding_perf.pdf'))

plt.figure()
plt.plot(num_neurons_list,reconstruction_perf)
plt.xlabel('Neurons')
plt.ylabel('Reconstruction accuracy')
plt.savefig(os.path.join(params['path_to_results'],'figures','optim_numNeurons_reconstruction_accuracy.pdf'))

plt.figure()
plt.plot(num_neurons_list,prediction_stats)
plt.xlabel('Neurons')
plt.ylabel('Decoding accuracy')
plt.savefig(os.path.join(params['path_to_results'],'figures','optim_numNeurons_decoding_accuracy.pdf'))

#%% Then select ideal number of neurons and optimize embedding dims
num_dims_list = [1,2,3,4,5,6,7,8]