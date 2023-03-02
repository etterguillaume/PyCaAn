from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
from numpy import diff
from numpy import std
from numpy import sqrt

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
    
    return interpolated_signal

def interpolate_1D(signal, behav_time, ca_time):
    interpolated_signal = np.zeros((len(ca_time),2))
    nan_vec = np.isnan(signal)
    if nan_vec is not None: # Remove nans to interpolate
        behav_time=behav_time[~nan_vec]
        signal=signal[~nan_vec]
    interp_func_x = interp1d(behav_time, signal, fill_value="extrapolate")
    interpolated_signal = interp_func_x(ca_time)   # use interpolation function returned by `interp1d`
    
    return interpolated_signal

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
    LT_direction[LT_direction>0] = 1.5
    LT_direction[LT_direction<=0] = .5

    return LT_direction

def compute_distance_time(interpolated_position, velocity, caTime, speed_threshold):
    elapsed_time = np.zeros(len(caTime))
    distance_travelled = np.zeros(len(caTime))
    time_counter=0
    distance_counter=0
    for i in range(1,len(velocity)):
        if (velocity[i] > speed_threshold): # Start of locomotor trajectory
            time_counter += caTime[i]-caTime[i-1]
            distance_counter += sqrt((interpolated_position[i,0]-interpolated_position[i-1,0])**2 + (interpolated_position[i,1]-interpolated_position[i-1,1])**2) # Euclidean distance

        if (velocity[i] <= speed_threshold): # End of locomotor trajectory
            time_counter = 0 # Reset counter
            distance_counter = 0

        elapsed_time[i] = time_counter
        distance_travelled[i] = distance_counter
    
    return elapsed_time, distance_travelled

def extract_tone(data, params):
    Fs = params['sampling_frequency']
    tone = data['tone']
    threshold = params['tone_threshold']

    # Init vectors 
    binary_signal = np.zeros_like(tone)

    binary_signal[tone>threshold]=1
    onset_signal = diff(binary_signal)
    onset_signal = np.append(onset_signal,0)
    flash_ts = np.where(onset_signal==1)[0]
    flash_int = diff(flash_ts/Fs)
    flash_int = np.append(flash_int,0)
    flash_freq = 1./flash_int

    # Create lists of transitions
    blank_to_slow=[]
    slow_to_fast=[]
    fast_to_solid=[]

    # First stim is obvious from deriv data
    blank_to_slow.append(0)

    for i in range(1,len(flash_freq)):
        if (flash_freq[i-1] < 4) & (flash_freq[i] > 4):
            blank_to_slow.append(i)
        elif (flash_freq[i-1] < 6) & (flash_freq[i] > 6):
            slow_to_fast.append(i)
        elif (flash_freq[i-1] > 6) & (flash_freq[i] < 4):
            fast_to_solid.append(i)

    # Assumed drop in frequency corresponding to solid light
    flash_ts = np.append(flash_ts, flash_ts[-1] + 1)
    fast_to_solid.append(len(flash_ts)-1)

    # Initialize state writes
    state = binary_signal*3+1
    state2write = 1

    frame_buffer = 10

    ct=0

    for i in range (len(state)-frame_buffer):
        if state2write == 1 and ct<len(blank_to_slow) and flash_ts[blank_to_slow[ct]]<i:
            blank_to_slow.pop(0)
        
        if state2write == 2 and ct<len(slow_to_fast) and flash_ts[slow_to_fast[ct]]<i:
            slow_to_fast.pop(0)
        
        if state2write == 3 and ct<len(fast_to_solid) and flash_ts[fast_to_solid[ct]]<i:
            fast_to_solid.pop(0)
        
        # Re-order state-transitions
        while len(slow_to_fast)>=(ct+frame_buffer) and slow_to_fast[ct]<=(blank_to_slow[ct]+frame_buffer):
            slow_to_fast.pop(0)

        while len(fast_to_solid)>=(ct+frame_buffer) and fast_to_solid[ct]<=(blank_to_slow[ct]+frame_buffer):
            fast_to_solid.pop(0)

        while len(fast_to_solid)>=(ct+frame_buffer) and fast_to_solid[ct]<=(slow_to_fast[ct]+frame_buffer):
            fast_to_solid.pop(0)

    
        # Detect state transitions
        if state2write==1 and ct<len(blank_to_slow) and flash_ts[blank_to_slow[ct]]==i:
            state2write = 2
        elif state2write==2 and ct<len(slow_to_fast) and flash_ts[slow_to_fast[ct]]==i:
            state2write = 3
        elif state2write==3 and ct<len(fast_to_solid) and flash_ts[fast_to_solid[ct]]==i:
            state2write = 4
        elif state2write==4 and state[i]==1:
            state2write = 1
            if ct+1<len(blank_to_slow) and ct+1<len(slow_to_fast) and ct+1<len(fast_to_solid):
                ct+=1
                while flash_ts[blank_to_slow[ct]] < i:
                    blank_to_slow.pop(0)

        state[i]=state2write

    data['seqLT_state'] = state

    return data

def preprocess_data(data, params):
    data['position'] = interpolate_2D(data['position'], data['behavTime'], data['caTime'])
    data['velocity'], data['running_ts'] = compute_velocity(data['position'], data['caTime'], params['speed_threshold'])
    data['elapsed_time'], data['distance_travelled'] = compute_distance_time(data['position'], 
                                                                             data['velocity'], 
                                                                             data['caTime'], 
                                                                             params['speed_threshold'])
    
    if data['task'] == 'LT' or data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
        data['LT_direction'] = extract_LT_direction(data['position'])
    
    # Interpolate tone if present
    if 'tone' in data:
        data['tone'] = interpolate_1D(data['tone'], data['behavTime'], data['caTime'])

    # Extract seqLT state if seqLT task
    if data['task'] == 'legoSeqLT':
        data = extract_tone(data,params)

    # Extract binary data
    data['binaryData'], data['neuralData'] = binarize_ca_traces(data['rawData'],
                                             z_threshold=params['z_threshold'],
                                             sampling_frequency=params['sampling_frequency']
                                            )

    return data