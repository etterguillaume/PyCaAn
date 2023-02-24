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
                    'pnrProj': np.array(f.get('ms/PeakToNoiseProj')),
                    'meanProj':np.array(f.get('ms/meanFrame')),
                    'experiment':np.array(f.get('ms/Experiment')),
                    'date':f.get('ms/dateNum')[0][0],
                    'SFPs':np.array(f.get('ms/SFPs')),
                    'caTime':np.array(f.get('ms/time'))[0]/1000, # convert ms->s
                    'rawData':np.array(f.get('ms/RawTraces')).T
                    }
                    )
    except: # If legacy MATLAB format
        f = sio.loadmat(path + '/ms.mat')
        data.update(
                    {
                    'corrProj':f['ms']['CorrProj'][0][0],
                    'pnrProj': f['ms']['PeakToNoiseProj'][0][0],
                    'meanProj': f['ms']['meanFrame'][0][0],
                    'experiment':f['ms']['Experiment'][0][0],
                    'date':f['ms']['dateNum'][0][0],
                    'SFPs':f['ms']['SFPs'][0][0],
                    'caTime':f['ms']['time'][0][0]/1000, # convert ms->s
                    'rawData':f['ms']['RawTraces'][0][0] 
                    }
                    )

    try: # If recent MATLAB format
        f = h5py.File(path + '/behav.mat','r')
        data.update(
                    {
                    'background':np.array(f.get('behav/background')),
                    'tone':np.array(f.get('behav/optosignal')),
                    'position':np.array(f.get('behav/position')).T,
                    'behavTime':np.array(f.get('behav/time'))[0]/1000, # convert ms->s
                    'mazeWidth_px':np.array(f.get('behav/width'))[0][0],
                    'mazeWidth_cm':np.array(f.get('behav/trackLength'))[0][0]
                    }
                    )
    except:
        f = sio.loadmat(path + '/behav.mat')
        data.update(
                    { # Note that older files do not have background/tones
                    'position':f['behav']['position'][0][0],
                    'behavTime':f['behav']['time'][0][0].T[0]/1000,
                    'mazeWidth_px':f['behav']['width'][0][0],
                    'mazeWidth_cm':f['behav']['trackLength'][0][0],
                    }
                    )

    return data