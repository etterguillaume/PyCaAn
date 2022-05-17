import h5py
import numpy as np

def load_data(path):
    data = {}
    f = h5py.File(path + '/behav.mat','r')
    data.update(
                {
                'corrProj':np.array(f.get('ms/CorrProj')),
                'experiment':np.array(f.get('ms/Experiment')),
                'SFPs':np.array(f.get('ms/SFPs')),
                'frameNum':np.array(f.get('ms/frameNum')),
                'numNeurons':np.array(f.get('ms/numNeurons')),
                'caTime':np.array(f.get('ms/time')),
                'caTrace':np.array(f.get('ms/RawTraces'))
                }
                )
    f = h5py.File(path + '/behav.mat','r')
    data.update(
                {
                'position':np.array(f.get('behav/position')),
                'behavTime':np.array(f.get('behav/time')),
                'mazeWidth_px':np.array(f.get('behav/width')),
                'mazeWidth_cm':np.array(f.get('behav/trackLength'))
                }
                )

    return data