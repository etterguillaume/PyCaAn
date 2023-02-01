import h5py
import numpy as np
import scipy.io as sio

def load_data(path):
    data = {}
    try: # If recent MATLAB format
        f = h5py.File(path + '/ms.mat','r')
        data.update(
                    {
                    'corrProj':np.array(f.get('ms/CorrProj')),
                    'experiment':np.array(f.get('ms/Experiment')),
                    'SFPs':np.array(f.get('ms/SFPs')),
                    'frameNum':np.array(f.get('ms/frameNum')[0]),
                    'numNeurons':int(np.array(f.get('ms/numNeurons'))[0]),
                    'caTime':np.array(f.get('ms/time'))[0]/1000, # convert ms->s
                    'caTrace':np.array(f.get('ms/RawTraces')).T
                    }
                    )
    except: # If legacy MATLAB format
        f = sio.loadmat(path + '/ms.mat')
        data.update(
                    {
                    'corrProj':f['ms']['CorrProj'][0][0],
                    'experiment':f['ms']['Experiment'][0][0],
                    'SFPs':f['ms']['SFPs'][0][0],
                    'frameNum':f['ms']['frameNum'][0][0],
                    'numNeurons':int(f['ms']['numNeurons'][0][0]),
                    'caTime':f['ms']['time'][0][0]/1000, # convert ms->s
                    'caTrace':f['ms']['RawTraces'][0][0] 
                    }
                    )

    try: # If recent MATLAB format
        f = h5py.File(path + '/behav.mat','r')
        data.update(
                    {
                    'position':np.array(f.get('behav/position')),
                    'behavTime':np.array(f.get('behav/time'))/1000, # convert ms->s
                    'mazeWidth_px':np.array(f.get('behav/width')),
                    'mazeWidth_cm':np.array(f.get('behav/trackLength'))
                    }
                    )
    except:
        f = sio.loadmat(path + '/behav.mat')
        data.update(
                    {
                    'position':f['behav']['position'][0][0],
                    'behavTime':f['behav']['time'][0][0].T[0]/1000,
                    'mazeWidth_px':f['behav']['width'][0][0],
                    'mazeWidth_cm':f['behav']['trackLength'][0][0],
                    }
                    )

    return data