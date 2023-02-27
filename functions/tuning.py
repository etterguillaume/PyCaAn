import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

def extract_2D_tuning(binarized_trace, interpolated_var, inclusion_ts, var_length, bin_size):
    X_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    Y_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    binarized_trace = binarized_trace[inclusion_ts]
    interpolated_var = interpolated_var[inclusion_ts]
    numFrames = len(binarized_trace)
    active_frames_in_bin = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1))
    occupancy_frames = np.zeros((len(Y_bin_vector)-1,len(X_bin_vector)-1))

    bin_vector = np.zeros(numFrames, dtype=int) # Vector that will specificy the bin# for each frame
    ct=0
    for y in range(len(Y_bin_vector)-1):
        for x in range(len(X_bin_vector)-1):
            frames_in_bin = (interpolated_var[:,0] >= X_bin_vector[x]) & (interpolated_var[:,0] < X_bin_vector[x+1]) & (interpolated_var[:,1] >= Y_bin_vector[y]) & (interpolated_var[:,1] < Y_bin_vector[y+1])
            occupancy_frames[y,x] = np.sum(frames_in_bin) # How many frames for that bin

            if frames_in_bin is not None: # if bin has been explored
                active_frames_in_bin[y,x] = np.sum(binarized_trace[frames_in_bin]) # Total number of frames of activity in that bin

            bin_vector[frames_in_bin] = ct

            ct+=1

    AMI = adjusted_mutual_info_score(binarized_trace,bin_vector)
    
    tuning_curve = active_frames_in_bin/occupancy_frames # Likelihood = number of active frames in bin/occupancy
    return AMI, occupancy_frames, active_frames_in_bin, tuning_curve

def extract_1D_tuning(binarized_trace, interpolated_var, inclusion_ts, var_length, bin_size):
    X_bin_vector = np.arange(0,var_length+bin_size,bin_size)
    binarized_trace = binarized_trace[inclusion_ts]
    interpolated_var = interpolated_var[inclusion_ts]
    numFrames = len(binarized_trace)
    active_frames_in_bin = np.zeros(len(X_bin_vector)-1)
    occupancy_frames = np.zeros(len(X_bin_vector)-1)

    bin_vector = np.zeros(numFrames, dtype=int) # Vector that will specificy the bin# for each frame
    ct=0

    for x in range(len(X_bin_vector)-1):
        frames_in_bin = (interpolated_var >= X_bin_vector[x]) & (interpolated_var < X_bin_vector[x+1])
        occupancy_frames[x] = np.sum(frames_in_bin) # How many frames for that bin

        if frames_in_bin is not None: # if bin has been explored
            active_frames_in_bin[x] = np.sum(binarized_trace[frames_in_bin]) # Total number of frames of activity in that bin

        bin_vector[frames_in_bin] = ct
        ct+=1

    AMI = adjusted_mutual_info_score(binarized_trace,bin_vector)
    
    tuning_curve = active_frames_in_bin/occupancy_frames # Likelihood = number of active frames in bin/occupancy
    return AMI, occupancy_frames, active_frames_in_bin, tuning_curve

# def extract_tuning_curves(data,params):
#     np.random.seed(params['seed'])
#     numFrames = data['binaryData'].shape[0]
#     numNeurons = data['binaryData'].shape[1] 
#     tuning_data={}
#     if data['task']=='OF':
#         MI_pvalue = np.zeros(numNeurons)
#         tuning_curves = np.zeros(numNeurons)
#         for cell_i in range(numNeurons): #TODO extract all at the same time
#             AMI, occupancy, tuning_curves = extract_2D_tuning(data['binaryData'][:,cell_i],
#                                                                                           data['position'],
#                                                                                           data['running_ts'],
#                                                                                           var_length=45,
#                                                                                           params=params['bin_size'])
            
#         tuning_data.update({
#             'spatial': {
#             'MI': MI,
#             'PI_pvalue': MI_pvalue,
#             'tuning_curves': tuning_curves,
#             'occupancy':
#             }
#         })

#     if data['task']=='legoOF':
#         spatial_tuning_data = extract_2D_tuning(data['binaryData'],
#                                         data['position'],
#                                         data['running_ts'],
#                                         var_length=50,
#                                         params=params['bin_size'])
#     if data['task']=='LT':
#         spatial_tuning_data = extract_1D_tuning(data['binaryData'],
#                                         data['position'],
#                                         data['running_ts'],
#                                         var_length=100,
#                                         params=params['bin_size'])
    
#     if data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
#         spatial_tuning_data = extract_1D_tuning(data['binaryData'],
#                                         data['position'],
#                                         data['running_ts'],
#                                         var_length=134,
#                                         params=params['bin_size'])
        
#     if data['task']=='legoToneLT':
#         #TODO tuning to tone vs reward?

#     if data['task']=='legoToneLT':
#         #TODO tuning to reward
#         #TODO tuning to time
#         #TODO tuning to tone

#     #TODO tuning to speed? time? direction?
    
#     return tuning_data