from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
from numpy import diff
from numpy import std
from numpy import sqrt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d, gaussian_filter

def normalize(data): # Normalize between 0-1
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def binarize_ca_traces(ca_traces, z_threshold, sampling_frequency): #TODO MAKE SURE THIS WORKS!!!!!!
    binarized_traces = np.zeros(ca_traces.shape, dtype='bool')
    neural_data = np.zeros(ca_traces.shape)
    b, a = signal.butter(20, 2/(sampling_frequency/2), 'low', analog=False)
    for trace_num in range(ca_traces.shape[1]):
        ca_trace = ca_traces[:,trace_num]
        filtered_trace = signal.detrend(signal.filtfilt(b,a,ca_trace)) # Detrend and filter
        norm_trace = filtered_trace/std(filtered_trace)
        d1_trace = diff(filtered_trace)
        d1_trace = np.append(d1_trace,0)
        binarized_traces[(norm_trace > z_threshold) & (d1_trace > 0),trace_num] = 1
        neural_data[:,trace_num] = norm_trace

    return binarized_traces, neural_data

def clean_timestamps(data):
    # CI data
    _, unique_timestamps = np.unique(data['caTime'],return_index=True) # Find unique timestamps
    data['caTime'] = data['caTime'][unique_timestamps]
    data['rawData'] = data['rawData'][unique_timestamps,:]

    # Behavior
    _, unique_timestamps = np.unique(data['behavTime'],return_index=True) # Find unique timestamps
    data['behavTime'] = data['behavTime'][unique_timestamps]
    data['position'] = data['position'][unique_timestamps,:]
    if 'tone' in data:
        data['tone'] = data['tone'][unique_timestamps]
    
    return data

def interpolate_2D(signal, behav_time, ca_time):
    interpolated_signal = np.zeros((len(ca_time),2))
    nan_vec = np.isnan(np.sum(signal,axis=1))
    if nan_vec is not None: # Remove nans to interpolate
        behav_time=behav_time[~nan_vec]
        signal=signal[~nan_vec]
    interp_func_x = interp1d(behav_time, signal[:,0], fill_value="extrapolate")
    interp_func_y = interp1d(behav_time, signal[:,1], fill_value="extrapolate")
    interpolated_signal[:,0] = interp_func_x(ca_time)   # use interpolation function returned by `interp1d`
    interpolated_signal[:,1] = interp_func_y(ca_time)   # use interpolation function returned by `interp1d`
    interpolated_signal[interpolated_signal<0]=0 # Remove negative values
    
    return interpolated_signal

def interpolate_1D(signal, behav_time, ca_time, kind='nearest'):
    interpolated_signal = np.zeros((len(ca_time),2))
    nan_vec = np.isnan(signal)
    if nan_vec is not None: # Remove nans to interpolate
        behav_time=behav_time[~nan_vec]
        signal=signal[~nan_vec]
    interp_func_x = interp1d(behav_time, signal, kind=kind, fill_value="extrapolate")
    interpolated_signal = interp_func_x(ca_time)   # use interpolation function returned by `interp1d`
    
    return interpolated_signal

def smooth_1D(signal, params):
    smoothed_signal = gaussian_filter1d(signal,sigma=params['smoothing_sigma'])
    return smoothed_signal

def smooth_2D(signal, params):
    smoothed_signal = gaussian_filter(signal,sigma=params['smoothing_sigma'])
    return smoothed_signal

def compute_velocity(interpolated_position, caTime, speed_threshold):
    velocity = np.zeros(interpolated_position.shape[0])
    running_ts = np.zeros(interpolated_position.shape[0], dtype='bool')
    for i in range(1,len(velocity)):
        velocity[i] = sqrt((interpolated_position[i,0]-interpolated_position[i-1,0])**2 + (interpolated_position[i,1]-interpolated_position[i-1,1])**2)/(caTime[i]-caTime[i-1])

    velocity = signal.savgol_filter(velocity, 5, 3)
    running_ts[velocity>speed_threshold] = True
    
    return velocity, running_ts

def extract_LT_direction(interpolated_position):
    LT_direction=diff(interpolated_position)
    LT_direction=np.append(LT_direction,0) # Add last missing datapoint
    LT_direction[LT_direction>0] = 1
    LT_direction[LT_direction<=0] = 0

    return LT_direction.astype(int)

def compute_heading(interpolated_position):
    heading = np.zeros(len(interpolated_position))
    heading = np.mod(np.arctan2(diff(interpolated_position[:,0]), diff(interpolated_position[:,1]))*180/np.pi, 360)
    heading=np.append(heading,np.nan)

    return heading

def compute_distance_time(interpolated_position, velocity, caTime, speed_threshold):
    elapsed_time = np.zeros(len(caTime))
    distance_travelled = np.zeros(len(caTime))
    time2stop = np.zeros(len(caTime))
    distance2stop = np.zeros(len(caTime))
    elapsed_time_counter=0
    travelled_distance_counter=0
    for i in range(1,len(velocity)):
        if (velocity[i] > speed_threshold): # Start of locomotor trajectory
            elapsed_time_counter += caTime[i]-caTime[i-1]
            travelled_distance_counter += sqrt((interpolated_position[i,0]-interpolated_position[i-1,0])**2 + (interpolated_position[i,1]-interpolated_position[i-1,1])**2) # Euclidean distance

        if (velocity[i] <= speed_threshold): # End of locomotor trajectory
            elapsed_time_counter = 0 # Reset counter
            travelled_distance_counter = 0

        elapsed_time[i] = elapsed_time_counter
        distance_travelled[i] = travelled_distance_counter

    # Time/distance to destination
    for i in reversed(range(1,len(velocity))):
        if (velocity[i] > speed_threshold): # Start of locomotor trajectory
            elapsed_time_counter += caTime[i]-caTime[i-1]
            travelled_distance_counter += sqrt((interpolated_position[i,0]-interpolated_position[i-1,0])**2 + (interpolated_position[i,1]-interpolated_position[i-1,1])**2) # Euclidean distance

        if (velocity[i] <= speed_threshold): # End of locomotor trajectory
            elapsed_time_counter = 0 # Reset counter
            travelled_distance_counter = 0

        time2stop[i] = elapsed_time_counter
        distance2stop[i] = travelled_distance_counter

    return elapsed_time, distance_travelled, time2stop, distance2stop

def extract_tone(data, params):
    data['binaryTone'] = data['tone']
    data['binaryTone'][data['tone']>=params['tone_threshold']]=1
    data['binaryTone'][data['tone']<params['tone_threshold']]=0
    data['binaryTone']=data['binaryTone'].astype(int)

    return data

def extract_seqLT_tone(data, params):
    tone = data['tone']
    state=np.zeros(tone.shape,dtype=int)
    window=params['sampling_frequency']*2
    half_window, remainder=divmod(window,2) # divide by 2, get remainder
    init_seg = tone[0:half_window] # pad by copying beggining
    end_seg = tone[-1-half_window+remainder:-1] # pad by copying end

    padded_tone = np.concatenate((init_seg,tone,end_seg))

    min_vec = np.zeros(len(tone))
    max_vec = np.zeros(len(tone))
    sum_vec = np.zeros(len(tone))
    median_vec = np.zeros(len(tone))

    for i in range(len(tone)):
        min_vec[i:i+window]=np.min(padded_tone[i:i+window])
        max_vec[i:i+window]=np.max(padded_tone[i:i+window])
        sum_vec[i:i+window]=np.sum(padded_tone[i:i+window])
        median_vec[i:i+window]=np.median(padded_tone[i:i+window])

    sum_minus_median_zvec = zscore(sum_vec-median_vec)
    min_minus_median_zvec = zscore(min_vec-median_vec)
    min_zvec = zscore(min_vec)
    max_zvec = zscore(max_vec)

    state2write=0

    for i in range(len(state)):
        if sum_minus_median_zvec[i]<min_minus_median_zvec[i] and min_zvec[i]>max_zvec[i]:
            # Blank
            state2write=0
        
        elif sum_minus_median_zvec[i]>min_minus_median_zvec[i] and min_zvec[i]>max_zvec[i] and state2write==2:
            # Fast
            state2write=3

        elif sum_minus_median_zvec[i]>min_minus_median_zvec[i] and min_zvec[i]<max_zvec[i] and state2write==0:
            # blank to slow
            state2write=1
            
        elif sum_minus_median_zvec[i]<min_minus_median_zvec[i] and min_zvec[i]<max_zvec[i] and state2write==1:
            # Slow to fast
            state2write=2

        state[i]=state2write

    data['seqLT_state'] = state

    return data

def preprocess_data(data, params):
    # Error checking
    assert len(data['behavTime'])==len(data['position']), 'behavTime and behavioral vector are not the same length'

    data = clean_timestamps(data) # only include unique timestamps
    data['position'] = interpolate_2D(data['position'], data['behavTime'], data['caTime'])
    data['velocity'], data['running_ts'] = compute_velocity(data['position'], data['caTime'], params['speed_threshold'])
    data['elapsed_time'], data['distance_travelled'], data['time2stop'], data['distance2stop'] = compute_distance_time(data['position'], 
                                                                             data['velocity'], 
                                                                             data['caTime'], 
                                                                             params['speed_threshold'])
    
    if data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
        data['LT_direction'] = extract_LT_direction(data['position'][:,0])

    else: #Compute heading
        data['heading'] = compute_heading(data['position'])

    # Interpolate tone if present
    if 'tone' in data:
        data['tone'] = interpolate_1D(data['tone'], data['behavTime'], data['caTime'],kind='nearest')

    # Extract binary data
    data['binaryData'], data['neuralData'] = binarize_ca_traces(data['rawData'],
                                             z_threshold=params['z_threshold'],
                                             sampling_frequency=params['sampling_frequency']
                                            )

    return data