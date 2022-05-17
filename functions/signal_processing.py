from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
from numpy import diff
from numpy import std
from numpy import sqrt

def binarize_ca_traces(ca_traces, z_threshold, sampling_frequency):
    binarized_traces = np.zeros(ca_traces.shape, dtype='bool')
    b, a = signal.butter(20, 2/(sampling_frequency/2), 'low', analog=False)
    for trace_num in range(ca_traces.shape[0]):
        ca_trace = ca_traces[trace_num,:]
        filtered_trace = signal.detrend(signal.filtfilt(b,a,ca_trace)) # Detrend and filter
        norm_trace = filtered_trace/std(filtered_trace)
        d1_trace = diff(filtered_trace)
        d1_trace = np.append(d1_trace,0)

        binarized_traces[trace_num,(norm_trace > z_threshold) & (d1_trace > 0)] = 1

    return binarized_traces

def interpolate_behavior(position, behav_time, ca_time):
    interpolated_position = np.zeros((2,ca_time.shape[1]))
    interp_func_x = interp1d(behav_time[0,:], position[0,:], fill_value="extrapolate")
    interp_func_y = interp1d(behav_time[0,:], position[1,:], fill_value="extrapolate")
    interpolated_position[0,:] = interp_func_x(ca_time[0,:])   # use interpolation function returned by `interp1d`
    interpolated_position[1,:] = interp_func_y(ca_time[0,:])   # use interpolation function returned by `interp1d`
    
    return interpolated_position

def compute_velocity(interpolated_position, caTime, speed_threshold):
    velocity = np.zeros(interpolated_position.shape[1])
    running_ts = np.zeros(interpolated_position.shape[1], dtype='bool')
    for i in range(1,len(velocity)):
        velocity[i] = sqrt((interpolated_position[0,i]-interpolated_position[0,i-1])**2 + (interpolated_position[1,i]-interpolated_position[1,i-1])**2)/(caTime[0,i]-caTime[0,i-1])

    velocity = signal.savgol_filter(velocity, 5, 3)
    running_ts[velocity>speed_threshold] = True
    
    return velocity, running_ts