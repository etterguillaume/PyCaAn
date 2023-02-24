import numpy as np

def extract_2D_tuning(binarized_trace, interpolated_position, inclusion_ts, var_length, bin_size):
    X_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    Y_bin_vector = np.arange(0,var_length+bin_size, bin_size)
    binarized_traces = binarized_traces[inclusion_ts]
    interpolated_position = interpolated_position[inclusion_ts]
    numNeurons, numFrames = binarized_traces.shape

    prob_being_active = np.sum(binarized_traces, axis=1)/numNeurons
    tuning_curve = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1))
    occupancy = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1))
    explored_map = np.ones((len(Y_bin_vector)-1,len(X_bin_vector)-1),dtype='bool')
    MI = 0

    # Compute occupancy
    for y in range(len(Y_bin_vector)-1):
        for x in range(len(X_bin_vector)-1):
            position_idx = (interpolated_position[0,:] >= X_bin_vector[x]) & (interpolated_position[0,:] < X_bin_vector[x+1]) & (interpolated_position[1,:] >= Y_bin_vector[y]) & (interpolated_position[1,:] < Y_bin_vector[y+1])
            occupancy[y,x] = np.sum(position_idx)
    explored_map[occupancy<1]=False


    for y in range(len(Y_bin_vector)-1):
        for x in range(len(X_bin_vector)-1):
            binarized_spatial_vector = np.zeros(numFrames,dtype='bool')
            position_idx = (interpolated_position[0,:] >= X_bin_vector[x] and interpolated_position[0,:] < X_bin_vector[x+1] and interpolated_position[1,:] >= Y_bin_vector[y] and interpolated_position[1,:] < Y_bin_vector[y+1])

            if position_idx: # if not empty
                binarized_spatial_vector[position_idx]=1
                
                # get min number of frames for a given bin
                # Use this number to sample activity in other bins randomly? Or taken care by shuffles?

                activity_in_bin_idx = (binarized_trace == 1) & (binarized_spatial_vector == 1)
                inactivity_in_bin_idx = (binarized_trace == 0) & (binarized_spatial_vector == 1)
                tuning_curve[y,x] = len(activity_in_bin_idx)/len(position_idx)
                
                joint_prob_active = len(activity_in_bin_idx)/len(binarized_trace)
                joint_prob_inactive = len(inactivity_in_bin_idx)/len(binarized_trace)
                prob_in_bin = len(position_idx)/len(binarized_trace);
                
                if joint_prob_active>0:
                    MI = MI + joint_prob_active*np.log2(joint_prob_active/(prob_in_bin*prob_being_active))
            
                if joint_prob_inactive>0:
                    MI = MI + joint_prob_inactive*np.log2(joint_prob_inactive/(prob_in_bin*(1-prob_being_active)))
            
    posterior = tuning_curve*occupancy/prob_being_active
    occupancy = occupancy/numFrames # convert from raw frame count into % occupancy
    
    #TODO smooth fields

    #MI, posterior, occupancy_map, prob_being_active, likelihood
    return MI, posterior, occupancy, prob_being_active, tuning_curve

def extract_1D_tuning(binarized_trace, interpolated_position, inclusion_ts, var_length, bin_size):
    return MI, posterior, occupancy, prob_being_active, tuning_curve

def extract_tuning_curves(data,params):
    np.random.seed(params['seed'])
    numFrames = data['binaryData'].shape[0]
    numNeurons = data['binaryData'].shape[1] 
    tuning_data={}
    if data['task']=='OF':
        MI_pvalue = np.zeros(numNeurons)
        tuning_curves = np.zeros(numNeurons)
        for cell_i in range(numNeurons): #TODO extract all at the same time
            MI, posterior, occupancy, prob_being_active, tuning_curves = extract_2D_tuning(data['binaryData'][:,cell_i],
                                                                                          data['position'],
                                                                                          data['running_ts'],
                                                                                          var_length=45,
                                                                                          params=params['bin_size'])
            shuffled_MI = np.zeros(params['num_surrogates'])
            for shuffle_i in range(params['num_surrogates']):
                random_ts = int(np.ceil(np.random.random()*numFrames))
                shuffled_trace = np.roll(data['binaryData'][:,cell_i],random_ts)
                shuffled_MI[shuffle_i], _, _, _, _ = extract_2D_tuning(data['binaryData'][:,cell_i],
                                                   data['position'],
                                                   data['running_ts'],
                                                   var_length=45,
                                                   params=params['bin_size'])
            MI_pvalue[cell_i] = sum(shuffled_MI>MI[cell_i])/params['num_surrogates']
        
        tuning_data.update({
            'spatial': {
            'MI': MI,
            'PI_pvalue': MI_pvalue,
            'tuning_curves': tuning_curves,
            'occupancy':
            }
        })

    if data['task']=='legoOF':
        spatial_tuning_data = extract_2D_tuning(data['binaryData'],
                                        data['position'],
                                        data['running_ts'],
                                        var_length=50,
                                        params=params['bin_size'])
    if data['task']=='LT':
        spatial_tuning_data = extract_1D_tuning(data['binaryData'],
                                        data['position'],
                                        data['running_ts'],
                                        var_length=100,
                                        params=params['bin_size'])
    
    if data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
        spatial_tuning_data = extract_1D_tuning(data['binaryData'],
                                        data['position'],
                                        data['running_ts'],
                                        var_length=134,
                                        params=params['bin_size'])
        
    if data['task']=='legoToneLT':
        #TODO tuning to tone vs reward?

    if data['task']=='legoToneLT':
        #TODO tuning to reward
        #TODO tuning to time
        #TODO tuning to tone

    #TODO tuning to speed? time? direction?
    
    return tuning_data