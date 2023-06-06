import numpy as np

def simulate_activity(recording_length, num_bins, ground_truth_info, sampling):
    assert num_bins>1
    if num_bins==2:
        variable = np.ones(recording_length)
    else:
        variable = np.random.choice(np.arange(1,num_bins),recording_length) # randomly sample bins
    activity = np.zeros(recording_length,dtype=bool)

    variable[np.random.choice(np.arange(recording_length), int(sampling*recording_length), replace=False)] = 0 # Set number of samples for which activity could predict the variable
    activity[variable==0]=True # Neural activity perfectly predicts variable

    bits2flip = np.random.choice(np.arange(recording_length), int(recording_length*(1-ground_truth_info)), replace=False)
    bits2flip = np.random.choice(bits2flip,int(0.5*len(bits2flip))) # Flip a coin to decide whether to flip those bits
    activity[bits2flip] = ~activity[bits2flip]

    return activity, variable