import h5py
import numpy as np
import scipy.io as sio
import os

def load_data(path):
    data = {}
    split_path = path.split(os.sep)
    split_name = split_path[-1].split('_')
    
    # Basic information
    data.update(
                {
                'path': path,
                'day': int(split_name[-1]),
                'task':split_name[1],
                'subject':split_path[-2],
                'region': split_path[-3],
                'sex': 'U',
                'age': 0,
                'condition': 'normal',
                'darkness': False,
                'optoStim': False,
                'rewards': True
                }
    )

    # Extra conditions
    if 'dark' in split_name:
        data['darkness'] = True
    if '8Hz' in split_name:
        data['optoStim'] = '8Hz'
    if 'scrambled' in split_name:
        data['optoStim'] = 'scrambled'
    if 'norewards' in split_name:
        data['rewards'] = False
    if 'AD' in split_name:
        data['condition'] = 'AD'

    try: # If recent MATLAB format
        f = h5py.File(path + '/ms.mat','r')
        data.update({
                    'corrProj':np.array(f.get('ms/CorrProj')),
                    'pnrProj': np.array(f.get('ms/PeakToNoiseProj')),
                    #'meanProj':np.array(f.get('ms/meanFrame')),
                    #'experiment':np.array(f.get('ms/Experiment')),
                    'SFPs':np.array(f.get('ms/SFPs')),
                    'caTime':np.array(f.get('ms/time'))[0]/1000, # convert ms->s
                    'rawData':np.array(f.get('ms/RawTraces')).T})
    except: # If legacy MATLAB format
        f = sio.loadmat(path + '/ms.mat')
        data.update(
                    {
                    'corrProj':f['ms']['CorrProj'][0][0],
                    'pnrProj': f['ms']['PeakToNoiseProj'][0][0],
                    #'meanProj': f['ms']['meanFrame'][0][0],
                    #'experiment':f['ms']['Experiment'][0][0],
                    'SFPs':f['ms']['SFPs'][0][0],
                    'caTime':f['ms']['time'][0][0].T[0]/1000, # convert ms->s
                    'rawData':f['ms']['RawTraces'][0][0] 
                    }
                    )

    try: # If recent MATLAB format
        f = h5py.File(path + '/behav.mat','r')
        data.update(
                    {
                    'position':np.array(f.get('behav/position')).T,
                    'behavTime':np.array(f.get('behav/time'))[0]/1000, # convert ms->s
                    'mazeWidth_px':np.array(f.get('behav/width'))[0][0],
                    'mazeWidth_cm':np.array(f.get('behav/trackLength'))[0][0]
                    }
                    )
        if 'background' in f['behav']:
            data.update({'background':np.array(f.get('behav/background'))})
        if 'optosignal' in f['behav']:
            data.update({'tone':np.array(f.get('behav/optosignal'))[:,0]})

    except:
        f = sio.loadmat(path + '/behav.mat')
        try: # If old format
            data.update(
                        { # Note that older files do not have background/tones
                        'position':f['behav']['position'][0][0],
                        'behavTime':f['behav']['time'][0][0].T[0]/1000,
                        'mazeWidth_px':f['behav']['width'][0][0],
                        'mazeWidth_cm':f['behav']['trackLength'][0][0],
                        }
                        )
        except: # else must be recent Deeplabcut output
            data.update(
                        { # Note that older files do not have background/tones
                        'position':f['behav']['ledPosition'][0][0], # Use LED to match older recordings
                        'headDirection':f['behav']['headDirection'][0][0][0],
                        'behavTime':f['behav']['time'][0][0][0]/1000,
                        'mazeWidth_cm':f['behav']['width'][0][0],
                        'mazeWidth_px':f['behav']['width'][0][0]/f['behav']['cmPerPixels'][0][0]
                        }
                        )

    return data