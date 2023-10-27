from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.neighbors import KNeighborsRegressor as knn_reg
from scipy.interpolate import griddata

import numpy as np
import warnings

def quantize_embedding(embedding, var, bin_vec):
        # Digitize labels
        quantized_var=np.digitize(var, bin_vec, right=False)

        # Quantize and align embeddings
        quantized_embedding = np.zeros((len(bin_vec),embedding.shape[1]))
        
        for i, bin_val in enumerate(bin_vec):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                quantized_embedding[i,:] = np.nanmean(embedding[quantized_var==i],axis=0)
        
        # Interpolate through missing values
        for dim in range(quantize_embedding.shape[1]):
             
        
        return quantized_embedding, quantized_var, bin_vec

def extract_hyperalignment_score(embedding_ref,
                                var_ref,
                                trainingFrames_ref,
                                testingFrames_ref,
                                embedding_pred,
                                var_pred,
                                trainingFrames_pred,
                                testingFrames_pred,
                                bin_vec):
    
    train_embedding_ref=embedding_ref[trainingFrames_ref]
    test_embedding_ref=embedding_ref[testingFrames_ref]
    train_embedding_pred=embedding_pred[trainingFrames_pred]
    test_embedding_pred=embedding_pred[testingFrames_pred]
    
    train_var_ref = var_ref[trainingFrames_ref]
    test_var_ref = var_ref[testingFrames_ref]
    train_var_pred = var_pred[trainingFrames_pred]
    test_var_pred = var_pred[testingFrames_pred]

    train_quantized_embedding_ref, _ = quantize_embedding(train_embedding_ref,
                                                            train_var_ref, 
                                                            bin_vec)
    test_quantized_embedding_ref, _ = quantize_embedding(test_embedding_ref,
                                                            test_var_ref,
                                                            bin_vec)
    train_quantized_embedding_pred, _ = quantize_embedding(train_embedding_pred,
                                                            train_var_ref,
                                                            bin_vec)
    test_quantized_embedding_pred, _ = quantize_embedding(test_embedding_pred,
                                                            test_var_ref,
                                                            bin_vec)
    
    # Train decoder
    decoder_AB = lin_reg().fit(train_quantized_embedding_ref, train_quantized_embedding_pred)
    decoder_BA = lin_reg().fit(train_quantized_embedding_pred, train_quantized_embedding_ref)
    
    # Assess reconstruction error
    HAS_AB = decoder_AB.score(test_quantized_embedding_ref, test_quantized_embedding_pred)
    HAS_BA = decoder_BA.score(test_quantized_embedding_pred, test_quantized_embedding_ref)

    return HAS_AB, HAS_BA