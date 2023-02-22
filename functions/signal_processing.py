from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
from numpy import diff
from numpy import std
from numpy import sqrt

def binarize_ca_traces(ca_traces, z_threshold, sampling_frequency): #TODO MAKE SURE THIS WORKS!!!!!!
    binarized_traces = np.zeros(ca_traces.shape, dtype='bool')
    filtnorm_traces = np.zeros(ca_traces.shape)
    b, a = signal.butter(20, 2/(sampling_frequency/2), 'low', analog=False)
    max_val = np.max(ca_traces.flatten())
    for trace_num in range(ca_traces.shape[1]):
        ca_trace = ca_traces[:,trace_num]
        filtered_trace = signal.detrend(signal.filtfilt(b,a,ca_trace)) # Detrend and filter
        norm_trace = filtered_trace/std(filtered_trace)
        d1_trace = diff(filtered_trace)
        d1_trace = np.append(d1_trace,0)
        binarized_traces[(norm_trace > z_threshold) & (d1_trace > 0),trace_num] = 1
        #filtnorm_traces[:,trace_num] = filtered_trace/max_val
        filtnorm_traces[:,trace_num] = filtered_trace

    return binarized_traces, filtnorm_traces

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
    #data['position'] = data['position']/data['mazeWidth_cm']
    #data['max_velocity'] = np.max(data['velocity'])
    #data['velocity'] = data['velocity']/data['max_velocity']

    data['binaryData'], data['neuralData'] = binarize_ca_traces(data['caTrace'],
                                             z_threshold=params['z_threshold'],
                                             sampling_frequency=params['sampling_frequency']
                                             )

    return data