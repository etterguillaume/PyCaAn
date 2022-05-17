import h5py
import numpy as np

def load_calcium_data(path):
    ms = {}
    f = h5py.File(path,'r')
    ms.update(
        {'corrProj':np.array(f.get('ms/CorrProj')),
        'experiment':np.array(f.get('ms/Experiment')),
        'SFPs':np.array(f.get('ms/SFPs')),
        'frameNum':np.array(f.get('ms/frameNum')),
        'numNeurons':np.array(f.get('ms/numNeurons')),
        'caTime':np.array(f.get('ms/time')),
        'caTrace':np.array(f.get('ms/RawTraces'))})
    return ms