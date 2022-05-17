from scipy.interpolate import interp1d
from scipy import signal
import numpy as np
from numpy import diff
from numpy import std
from scipy.stats import zscore

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
    interp_func = interp1d(behav_time[:,0], position[:,0], fill_value="extrapolate")
    interpolated_position = interp_func(ca_time[:,0])   # use interpolation function returned by `interp1d`
    
    return interpolated_position