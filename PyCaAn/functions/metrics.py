import numpy as np
from numpy import sqrt
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

def analyze_binary_reconstruction(params, model, data_loader):
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

def extract_binary_stats(binary_data):
    numFrames, numNeurons = binary_data.shape
    trans_probability = []
    prob_being_active = np.zeros(numNeurons)

    for i in range(numNeurons):
        prob_being_active[i] = np.sum(binary_data[:,i])/numFrames
        
    return prob_being_active, trans_probability

def extract_total_distance_travelled(interpolated_position):
    total_distance_travelled = 0
    for i in range(1,len(interpolated_position)):
        total_distance_travelled += sqrt((interpolated_position[i,0]-interpolated_position[i-1,0])**2 + (interpolated_position[i,1]-interpolated_position[i-1,1])**2) # Euclidean distance
    return total_distance_travelled