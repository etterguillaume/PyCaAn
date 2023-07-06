from sklearn.linear_model import LinearRegression as lin_reg
import numpy as np
import warnings

def extract_hyperalignment_score(embedding_ref,
                                var_ref,
                                trainingFrames_ref,
                                testingFrames_ref,
                                embedding_pred,
                                var_pred,
                                trainingFrames_pred,
                                testingFrames_pred,
                                bin_vec):
    # TODO separate fit score (on training set) and forward transfer (on test set)
    embedding_ref=embedding_ref[trainingFrames_ref]
    embedding_pred=embedding_pred[trainingFrames_pred]
    var_ref = var_ref[trainingFrames_ref]
    var_pred = var_pred[trainingFrames_pred]

    # Digitize labels
    quantized_var_ref=np.digitize(var_ref, bin_vec, right=False)
    quantized_var_pred=np.digitize(var_pred, bin_vec, right=False)

    # Quantize and align embeddings
    quantized_embedding_ref = np.zeros((len(bin_vec),embedding_ref.shape[1]))
    quantized_embedding_pred = np.zeros((len(bin_vec),embedding_pred.shape[1]))
    
    to_delete = []
    for i, _ in enumerate(bin_vec):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            quantized_embedding_ref[i,:] = np.nanmean(embedding_ref[quantized_var_ref==i],axis=0)
            quantized_embedding_pred[i,:] = np.nanmean(embedding_pred[quantized_var_pred==i],axis=0)
        
        # Remove nans or missing values
        if np.isnan(sum(quantized_embedding_ref[i])) or np.isnan(sum(quantized_embedding_pred[i])):
            to_delete.append(i)

    quantized_embedding_ref = np.delete(quantized_embedding_ref, to_delete, axis=0)
    quantized_embedding_pred = np.delete(quantized_embedding_pred, to_delete, axis=0)
    bin_vec = np.delete(bin_vec,to_delete, axis=0)

    # Normalize embeddings
    # (theoretically useless)

    # Train decoder
    decoder_AB = lin_reg().fit(quantized_embedding_ref, quantized_embedding_pred)
    decoder_BA = lin_reg().fit(quantized_embedding_pred, quantized_embedding_ref)
    
    # Assess reconstruction error
    HAS_AB = decoder_AB.score(quantized_embedding_ref, quantized_embedding_pred)
    HAS_BA = decoder_BA.score(quantized_embedding_pred, quantized_embedding_ref)

    return HAS_AB, HAS_BA, quantized_embedding_ref, quantized_embedding_pred, quantized_var_ref, quantized_var_pred, bin_vec