import numpy as np
import random
#  1D tuning curves


# 2D tuning
def extract_2D_tuning(binarized_traces, interpolated_position, inclusion_ts, params):
    X_bin_vector = np.arange(0,params['maze_width']+params['bin_size'],params['bin_size'])
    Y_bin_vector = np.arange(0,params['maze_width']+params['bin_size'],params['bin_size'])
    binarized_traces = binarized_traces[:,inclusion_ts]
    interpolated_position = interpolated_position[:,inclusion_ts]
    numNeurons, numFrames = binarized_traces.shape

    prob_being_active = np.sum(binarized_traces, axis=1)/numNeurons
    likelihood = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1))
    occupancy_map = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1))
    explored_map = np.ones((len(Y_bin_vector)-1,len(X_bin_vector)-1),dtype='bool')
    MI = 0

    # Compute occupancy
    for y in range(len(Y_bin_vector)-1):
        for x in range(len(X_bin_vector)-1):
            position_idx = (interpolated_position[0,:] >= X_bin_vector[x]) & (interpolated_position[0,:] < X_bin_vector[x+1]) & (interpolated_position[1,:] >= Y_bin_vector[y]) & (interpolated_position[1,:] < Y_bin_vector[y+1])
            occupancy_map[y,x] = np.sum(position_idx)
    explored_map[occupancy_map<1]=False

    for neuron in range(numNeurons):
        for y in range(len(Y_bin_vector)-1):
            for x in range(len(X_bin_vector)-1):
                binarized_spatial_vector = np.zeros(numFrames,dtype='bool')
                position_idx = (interpolated_position[0,:] >= X_bin_vector[x] and interpolated_position[0,:] < X_bin_vector[x+1] and interpolated_position[1,:] >= Y_bin_vector[y] and interpolated_position[1,:] < Y_bin_vector[y+1])

                if position_idx: # if not empty
                    binarized_spatial_vector[position_idx]=1
                    
                    # get min number of frames for a given bin
                    # Use this number to sample activity in other bins randomly

                    activity_in_bin_idx = (binarized_trace == 1) & (binarized_spatial_vector == 1)
                    inactivity_in_bin_idx = (binarized_trace == 0) & (binarized_spatial_vector == 1)
                    likelihood[y,x] = len(activity_in_bin_idx)/len(position_idx);
                    
                    joint_prob_active = len(activity_in_bin_idx)/len(binarized_trace);
                    joint_prob_inactive = len(inactivity_in_bin_idx)/len(binarized_trace);
                    prob_in_bin = len(position_idx)/len(binarized_trace);
                    
                    if joint_prob_active>0:
                        MI = MI + joint_prob_active*np.log2(joint_prob_active/(prob_in_bin*prob_being_active));
                
                    if joint_prob_inactive>0:
                        MI = MI + joint_prob_inactive*np.log2(joint_prob_inactive/(prob_in_bin*(1-prob_being_active)));
            

    posterior = likelihood*occupancy_map/prob_being_active
    occupancy_map = occupancy_map/numFrames # convert from raw frame count into % occupancy
    
    
    # smooth fields
    # compute stability
    # compute significance

    
    
    
    #MI, posterior, occupancy_map, prob_being_active, likelihood
    return tuning_data