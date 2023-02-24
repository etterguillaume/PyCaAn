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

def interpolate_behavior(position, behav_time, ca_time):
    interpolated_position = np.zeros((len(ca_time),2))
    interp_func_x = interp1d(behav_time, position[:,0], fill_value="extrapolate")
    interp_func_y = interp1d(behav_time, position[:,1], fill_value="extrapolate")
    interpolated_position[:,0] = interp_func_x(ca_time)   # use interpolation function returned by `interp1d`
    interpolated_position[:,1] = interp_func_y(ca_time)   # use interpolation function returned by `interp1d`
    
    return interpolated_position

def compute_velocity(interpolated_position, caTime, speed_threshold):
    velocity = np.zeros(interpolated_position.shape[0])
    running_ts = np.zeros(interpolated_position.shape[0], dtype='bool')
    for i in range(1,len(velocity)):
        velocity[i] = sqrt((interpolated_position[i,0]-interpolated_position[i-1,0])**2 + (interpolated_position[i,1]-interpolated_position[i-1,1])**2)/(caTime[i]-caTime[i-1])

    velocity = signal.savgol_filter(velocity, 5, 3)
    running_ts[velocity>speed_threshold] = True
    
    return velocity, running_ts

def preprocess_data(data, params):
    data['position'] = interpolate_behavior(data['position'], data['behavTime'], data['caTime'])
    data['velocity'], data['running_ts'] = compute_velocity(data['position'], data['caTime'], params['speed_threshold'])
    
    # Normalize values
    data['normPosition'] = normalize(data['position'])
    data['normVelocity'] = normalize(data['velocity'])

    data['binaryData'], data['neuralData'] = binarize_ca_traces(data['caTrace'],
                                             z_threshold=params['z_threshold'],
                                             sampling_frequency=params['sampling_frequency']
                                             )

    return data

def extract_tone(data, params):
    threshold = 0.2
    Fs = params['sampling_frequency']
    tone = data['tone']

    # Init vectors 
    binary_signal = np.zeros(len(tone))

    binary_signal[tone>threshold]=1
    onset_signal = diff(binary_signal)
    onset_signal = np.append(onset_signal,0)
    flash_ts = np.where(onset_signal==1)
    flash_int = diff(flash_ts/Fs)
    flash_int = np.append(flash_int,0)
    flash_freq = 1/flash_int

    # Create lists of transitions
    blank_to_slow=[]
    slow_to_fast=[]
    fast_to_solid=[]

    # First stim is obvious from deriv data
    blank_to_slow.append(0)

    for i in range(1,len(flash_freq)):
        if flash_freq[i-1] < 4 & flash_freq[i] > 4:
            blank_to_slow.append(i)
        elif flash_freq[i-1] < 6 & flash_freq[i] > 6:
            slow_to_fast.append[i]
        elif flash_freq[i-1] > 6 & flash_freq[i] < 4:
            fast_to_solid.append(i)


    # Assumed drop in frequency corresponding to solid light
    flash_ts = np.append(flash_ts,flash_ts[-1])
    fast_to_solid = np.append(fast_to_solid, len(flash_ts))

    # Initialize state writes
    state = binary_signal*3+1
    state2write = 1

    frame_buffer = 10

    ct=0

    for i in range (len(state)-frame_buffer):
        if (state2write == 1) & (ct<=len(blank_to_slow)) & (flash_ts[blank_to_slow[ct]]<i):
            blank_to_slow.pop(0)
        
        if (state2write == 2) & (ct<=len(slow_to_fast)) & (flash_ts[slow_to_fast[ct]]<i):
            slow_to_fast.pop(0)
        
        if (state2write == 3) & (ct<=len(fast_to_solid)) & (flash_ts[fast_to_solid[ct]]<i):
            fast_to_solid.pop(0)
        
        # Re-order state-transitions
        while (len(slow_to_fast)>=(ct+frame_buffer)) & (slow_to_fast[ct]<=(blank_to_slow[ct]+frame_buffer)):
            slow_to_fast.pop(0)

        while (len(fast_to_solid)>=(ct+frame_buffer)) & (fast_to_solid[ct]<=(blank_to_slow[ct]+frame_buffer)):
            fast_to_solid.pop(0)

        while (len(fast_to_solid)>=(ct+frame_buffer)) & (fast_to_solid[ct]<=(slow_to_fast[ct]+frame_buffer)):
            fast_to_solid.pop(0)

    
        # Detect state transitions
    if (state2write==1) & (ct<=len(blank_to_slow)) & (flash_ts(blank_to_slow[ct])==i):
        state2write = 2
    elif (state2write==2) & (ct<=len(slow_to_fast)) & (flash_ts(slow_to_fast[ct])==i):
        state2write = 3
    elif (state2write==3) & (ct<=len(fast_to_solid)) & (flash_ts(fast_to_solid[ct])==i):
        state2write = 4
    elif (state2write==4) & (state(i)==1):
        state2write = 1
        if (ct+1<=len(blank_to_slow)) & (ct+1<=len(slow_to_fast)) & (ct+1<=len(fast_to_solid)):
            ct+=1
            while (flash_ts(blank_to_slow[ct]) < i):
                blank_to_slow.pop(0)

    state[i]=state2write

    data['seqLT_state'] = state

    return data