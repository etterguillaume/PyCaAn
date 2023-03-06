import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Ignore zero divide warnings
from sklearn.metrics import adjusted_mutual_info_score

def extract_2D_tuning(binaryData, interpolated_var, inclusion_ts, var_length, bin_size):
    X_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    Y_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    binaryData = binaryData[inclusion_ts]
    interpolated_var = interpolated_var[inclusion_ts]
    numFrames, numNeurons = binaryData.shape
    active_frames_in_bin = np.zeros((numNeurons,len(Y_bin_vector)-1,len(X_bin_vector)-1), dtype=int)
    occupancy_frames = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1), dtype=int)
    AMI = np.zeros(numNeurons)

    # Compute occupancy
    bin_vector = np.zeros(numFrames, dtype=int) # Vector that will specificy the bin# for each frame
    ct=0
    for y in range(len(Y_bin_vector)-1):
        for x in range(len(X_bin_vector)-1):
            frames_in_bin = (interpolated_var[:,0] >= X_bin_vector[x]) & (interpolated_var[:,0] < X_bin_vector[x+1]) & (interpolated_var[:,1] >= Y_bin_vector[y]) & (interpolated_var[:,1] < Y_bin_vector[y+1])
            occupancy_frames[y,x] = np.sum(frames_in_bin) # How many frames for that bin
            bin_vector[frames_in_bin] = ct
            ct+=1

    # Bin activity
    for neuron in range(numNeurons):
        for y in range(len(Y_bin_vector)-1):
            for x in range(len(X_bin_vector)-1):
                frames_in_bin = (interpolated_var[:,0] >= X_bin_vector[x]) & (interpolated_var[:,0] < X_bin_vector[x+1]) & (interpolated_var[:,1] >= Y_bin_vector[y]) & (interpolated_var[:,1] < Y_bin_vector[y+1])
                if frames_in_bin is not None: # if bin has been explored
                    active_frames_in_bin[neuron,y,x] = np.sum(binaryData[frames_in_bin,neuron]) # Total number of frames of activity in that bin

        AMI[neuron] = adjusted_mutual_info_score(binaryData[:,neuron],bin_vector)
    
    tuning_curve = active_frames_in_bin/occupancy_frames # Likelihood = number of active frames in bin/occupancy
    return AMI, occupancy_frames, active_frames_in_bin, tuning_curve

def extract_1D_tuning(binaryData, interpolated_var, inclusion_ts, var_length, bin_size):
    X_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    binaryData = binaryData[inclusion_ts]
    interpolated_var = interpolated_var[inclusion_ts]
    numFrames, numNeurons = binaryData.shape
    active_frames_in_bin = np.zeros((numNeurons,len(X_bin_vector)-1), dtype=int)
    occupancy_frames = np.zeros(len(X_bin_vector)-1, dtype=int)
    AMI = np.zeros(numNeurons)

    # Compute occupancy
    bin_vector = np.zeros(numFrames, dtype=int) # Vector that will specificy the bin# for each frame
    ct=0
    for x in range(len(X_bin_vector)-1):
        frames_in_bin = (interpolated_var >= X_bin_vector[x]) & (interpolated_var < X_bin_vector[x+1])
        occupancy_frames[x] = np.sum(frames_in_bin) # How many frames for that bin
        bin_vector[frames_in_bin] = ct
        ct+=1

    # Bin activity
    for neuron in range(numNeurons):
        for x in range(len(X_bin_vector)-1):
            frames_in_bin = (interpolated_var >= X_bin_vector[x]) & (interpolated_var < X_bin_vector[x+1])
            if frames_in_bin is not None: # if bin has been explored
                active_frames_in_bin[neuron,x] = np.sum(binaryData[frames_in_bin,neuron]) # Total number of frames of activity in that bin

        AMI[neuron] = adjusted_mutual_info_score(binaryData[:,neuron],bin_vector)
    
    tuning_curve = active_frames_in_bin/occupancy_frames # Likelihood = number of active frames in bin/occupancy
    return AMI, occupancy_frames, active_frames_in_bin, tuning_curve


def extract_discrete_tuning(binaryData, interpolated_var, inclusion_ts, var_length):
    discrete_bin_vector = np.arange(var_length)
    binaryData = binaryData[inclusion_ts]
    interpolated_var = interpolated_var[inclusion_ts]
    numFrames, numNeurons = binaryData.shape
    active_frames_in_bin = np.zeros((numNeurons,len(discrete_bin_vector)), dtype=int)
    occupancy_frames = np.zeros(len(discrete_bin_vector), dtype=int)
    AMI = np.zeros(numNeurons)

    # Compute occupancy
    bin_vector = np.zeros(numFrames, dtype=int) # Vector that will specificy the bin# for each frame
    ct=0
    for x in discrete_bin_vector:
        frames_in_bin = np.where(discrete_bin_vector=x)[0]
        occupancy_frames[x] = np.sum(frames_in_bin) # How many frames for that bin
        bin_vector[frames_in_bin] = ct
        ct+=1

    # Bin activity
    for neuron in range(numNeurons):
        for x in discrete_bin_vector:
            frames_in_bin = np.where(discrete_bin_vector=x)[0]
            if frames_in_bin is not None: # if bin has been explored
                active_frames_in_bin[neuron,x] = np.sum(binaryData[frames_in_bin,neuron]) # Total number of frames of activity in that bin

        AMI[neuron] = adjusted_mutual_info_score(binaryData[:,neuron],bin_vector)
    
    tuning_curve = active_frames_in_bin/occupancy_frames # Likelihood = number of active frames in bin/occupancy
    return AMI, occupancy_frames, active_frames_in_bin, tuning_curve