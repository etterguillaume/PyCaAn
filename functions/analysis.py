import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr as corr

def reconstruction_binary_accuracy(reconstruction, original):
    # Compute accuracy
    reconstruction = reconstruction.round()
    accuracy = sum(original==reconstruction)/len(original)
    true_positives = sum((reconstruction>0) & (original>0))
    false_positives = sum((reconstruction>0) & (original==0))
    false_negatives = sum((reconstruction==0) & (original>0))
    
    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    
    # Compute F-score
    F1 = 2*true_positives/(2*true_positives + false_positives + false_negatives)

    return accuracy, precision, recall, F1 

def analyze_AE_reconstruction(params, model, data_loader):
    device = torch.device(params['device'])
    total_inputs = np.empty((0,params['input_neurons']))
    total_reconstructions = np.empty((0,params['input_neurons']))

    for i, (x, _, _) in enumerate(data_loader):
        x = x.to(device)
        with torch.no_grad():
            reconstruction, _ = model(x)

            total_inputs = np.append(total_inputs, x.view(-1,params['input_neurons']), axis=0)
            total_reconstructions = np.append(total_reconstructions, reconstruction.view(-1,params['input_neurons']), axis=0)
    
    accuracy, precision, recall, F1 = reconstruction_binary_accuracy(total_reconstructions, total_inputs)

    return accuracy, precision, recall, F1 

def analyze_decoding(params, model, decoder, data_loader):
    total_predictions = np.empty((0,2)) # For x and y position
    total_positions = np.empty((0,2)) # For x and y position
    for i, (x, position, _) in enumerate(data_loader):
        device = torch.device(params['device'])
        x = x.to(device)
        with torch.no_grad():
            _, embedding = model(x)
            pred = decoder(embedding)

        total_positions = np.append(total_positions, position.view(-1,2), axis=0)
        total_predictions = np.append(total_predictions, pred.view(-1,2), axis=0)

    decoding_error = np.linalg.norm(total_predictions-total_positions) # Euclidean distance

    decoder_stats = corr(total_positions.flatten(),total_predictions.flatten())

    return decoding_error, decoder_stats