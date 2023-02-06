import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr as corr

def analyze_AE_reconstruction(params, model, data_loader):
    device = torch.device(params['device'])
    total_inputs = np.empty((0,params['input_neurons']))
    total_reconstructions = np.empty((0,params['input_neurons']))

    for i, (x, _, _) in enumerate(data_loader):
        x = x.to(device)
        with torch.no_grad():
            reconstruction, _ = model(x)

            total_inputs = np.append(total_inputs, x, axis=0)
            total_reconstructions = np.append(total_reconstructions, reconstruction, axis=0)

            total_reconstructions[total_reconstructions>.5] = 1
            total_reconstructions[total_reconstructions<=.5] = 0

            # Compute accuracy
            accuracy = sum(total_inputs==total_reconstructions)/len(total_inputs)
            true_positives = sum(total_reconstructions==1 & total_inputs==1)
            false_positives = sum(total_reconstructions==1 & total_inputs==0)
            false_negatives = sum(total_reconstructions==0 & total_inputs==1)
            
            precision = true_positives/(true_positives+false_positives)
            recall = true_positives/(true_positives+false_negatives)
            
            # Compute F-score
            F1 = 2*true_positives/(2*true_positives + false_positives + false_negatives)

    return accuracy, precision, recall, F1 

def analyze_decoding(params, model, decoder, data_loader):
    for i, (x, position, _) in enumerate(data_loader):
        device = torch.device(params['device'])
        x = x.to(device)
        with torch.no_grad():
            _, embedding = model(x)
            pred = decoder(embedding)

    total_predictions = np.empty(0)
    total_positions = np.empty(0)
    total_positions = np.append(total_positions, position[:,0], axis=0)
    total_predictions = np.append(total_predictions, pred[0], axis=0)

    decoding_error = np.abs(total_predictions-total_predictions) # Euclidean distance for 1D TODO: do 2D

    decoder_stats = corr(total_positions.flatten(),total_predictions.flatten())

    return decoding_error, decoder_stats