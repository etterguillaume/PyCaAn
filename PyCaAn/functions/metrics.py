import numpy as np
from numpy import sqrt

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

def extract_firing_properties(binaryData):
    numFrames, numNeurons = binaryData.shape
    marginal_likelihood = np.zeros(numNeurons)
    trans_prob = np.zeros(numNeurons)

    for neuron in range(numNeurons):
        marginal_likelihood[neuron] = np.sum(binaryData[:,neuron])/numFrames
        trans_prob[neuron] = sum(np.diff(binaryData[:,neuron].astype('int'))>0)/(numFrames-1)

    return marginal_likelihood, trans_prob