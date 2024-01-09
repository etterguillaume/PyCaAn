#%% Imports
import yaml
import os
from tqdm import tqdm
import numpy as np
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.functions.tuning import assess_covariate
from pycaan.functions.metrics import extract_total_distance_travelled
from argparse import ArgumentParser
import h5py

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--session_path', type=str, default='')
    args = parser.parse_args()
    return args

def extract_covariates_session(data, params):
    if not os.path.exists(params['path_to_results']):
        os.mkdir(params['path_to_results'])
    working_directory=os.path.join( 
        params['path_to_results'],
        f"{data['region']}_{data['subject']}_{data['task']}_{data['day']}" 
        )
    if not os.path.exists(working_directory): # If folder does not exist, create it
        os.mkdir(working_directory)

    if not os.path.exists(os.path.join(working_directory,'covariates.h5')) or params['overwrite_mode']=='always':
        # Pre-allocate data for covariates
        info_matrix = np.ones((5,5))
        pvalue_matrix = np.zeros((5,5))
        correlation_matrix = np.ones((5,5))
        labels=['space','time','distance','speed','heading']

        position=data['position']
        if data['task'] == 'legoOF':
            mazeSize=50
        elif data['task'] == 'plexiOF':
            mazeSize=49
        else:
            mazeSize=45

        if data['task'] == 'legoLT' or data['task'] == 'legoToneLT' or data['task'] == 'legoSeqLT':
            mazeSize=134
            position=data['position'][:,0]
        else:
            mazeSize=100
            position=data['position'][:,0]

        # Location vs time
        info_matrix[0,1], pvalue_matrix[0,1], correlation_matrix[0,1] = assess_covariate(
                position,
                data['elapsed_time'],
                data['running_ts'],
                mazeSize,
                params['spatialBinSize'],
                params['max_temporal_length'],
                params['temporalBinSize']
                )
        info_matrix[1,0], pvalue_matrix[1,0], correlation_matrix[1,0] = info_matrix[0,1], pvalue_matrix[0,1], correlation_matrix[0,1]

        # Location vs distance
        info_matrix[0,2], pvalue_matrix[0,2], correlation_matrix[0,2] = assess_covariate(
                position,
                data['distance_travelled'],
                data['running_ts'],
                mazeSize,
                params['spatialBinSize'],
                params['max_distance_length'],
                params['distanceBinSize']
                )
        info_matrix[2,0], pvalue_matrix[2,0], correlation_matrix[2,0] = info_matrix[0,2], pvalue_matrix[0,2], correlation_matrix[0,2]

        # Location vs velocity
        info_matrix[0,3], pvalue_matrix[0,3], correlation_matrix[0,3] = assess_covariate(
                position,
                data['velocity'],
                data['running_ts'],
                mazeSize,
                params['spatialBinSize'],
                params['max_velocity_length'],
                params['velocityBinSize']
                )
        info_matrix[3,0], pvalue_matrix[3,0], correlation_matrix[3,0] = info_matrix[0,3], pvalue_matrix[0,3], correlation_matrix[0,3]

        # Location vs velocity
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[0,4], pvalue_matrix[0,4], correlation_matrix[0,4] = assess_covariate(
                    position,
                    data['LT_direction'],
                    data['running_ts'],
                    mazeSize,
                    params['spatialBinSize'],
                    2,
                    1
                    )
            info_matrix[4,0], pvalue_matrix[4,0], correlation_matrix[4,0] = info_matrix[0,4], pvalue_matrix[0,4], correlation_matrix[0,4]

        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[0,4], pvalue_matrix[0,4], correlation_matrix[0,4] = assess_covariate(
                    position,
                    data['heading'],
                    data['running_ts'],
                    mazeSize,
                    params['spatialBinSize'],
                    360,
                    params['directionBinSize']
                    )
            info_matrix[4,0], pvalue_matrix[4,0], correlation_matrix[4,0] = info_matrix[0,4], pvalue_matrix[0,4], correlation_matrix[0,4]

        # Time vs distance
        info_matrix[1,2], pvalue_matrix[1,2], correlation_matrix[1,2] = assess_covariate(data['elapsed_time'],
                            data['distance_travelled'],
                            data['running_ts'],
                            params['max_temporal_length'],
                            params['temporalBinSize'],
                            params['max_distance_length'],
                            params['distanceBinSize'])
        info_matrix[2,1], pvalue_matrix[2,1], correlation_matrix[2,1] = info_matrix[1,2], pvalue_matrix[1,2], correlation_matrix[1,2]

        # Time vs speed
        info_matrix[1,3], pvalue_matrix[1,3], correlation_matrix[1,3] = assess_covariate(data['elapsed_time'],
                            data['velocity'],
                            data['running_ts'],
                            params['max_temporal_length'],
                            params['temporalBinSize'],
                            params['max_velocity_length'],
                            params['velocityBinSize'])
        info_matrix[3,1], pvalue_matrix[3,1], correlation_matrix[3,1] = info_matrix[1,3], pvalue_matrix[1,3], correlation_matrix[1,3]

        # Time vs heading
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[1,4], pvalue_matrix[1,4], correlation_matrix[1,4] = assess_covariate(data['elapsed_time'],
                                data['LT_direction'],
                                data['running_ts'],
                                params['max_temporal_length'],
                                params['temporalBinSize'],
                                2,
                                1
                                )
            info_matrix[4,1], pvalue_matrix[4,1], correlation_matrix[4,1] = info_matrix[1,4], pvalue_matrix[1,4], correlation_matrix[1,4]

        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[1,4], pvalue_matrix[1,4], correlation_matrix[1,4] = assess_covariate(data['elapsed_time'],
                                data['heading'],
                                data['running_ts'],
                                params['max_temporal_length'],
                                params['temporalBinSize'],
                                360,
                                params['directionBinSize']
                                )
            info_matrix[4,1], pvalue_matrix[4,1], correlation_matrix[4,1] = info_matrix[1,4], pvalue_matrix[1,4], correlation_matrix[1,4]

        # Distance vs speed
        info_matrix[2,3], pvalue_matrix[2,3], correlation_matrix[2,3] = assess_covariate(
                            data['distance_travelled'],
                            data['velocity'],
                            data['running_ts'],
                            params['max_distance_length'],
                            params['distanceBinSize'],
                            params['max_velocity_length'],
                            params['velocityBinSize'])
        info_matrix[3,2], pvalue_matrix[3,2], correlation_matrix[3,2] = info_matrix[2,3], pvalue_matrix[2,3], correlation_matrix[2,3]

        # distance vs heading
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[2,4], pvalue_matrix[2,4], correlation_matrix[2,4] = assess_covariate(
                                data['distance_travelled'],
                                data['LT_direction'],
                                data['running_ts'],
                                params['max_distance_length'],
                                params['distanceBinSize'],
                                2,
                                1
                                )
            info_matrix[4,2], pvalue_matrix[4,2], correlation_matrix[4,2] = info_matrix[2,4], pvalue_matrix[2,4], correlation_matrix[2,4]

        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[2,4], pvalue_matrix[2,4], correlation_matrix[2,4] = assess_covariate(
                                data['distance_travelled'],
                                data['heading'],
                                data['running_ts'],
                                params['max_distance_length'],
                                params['distanceBinSize'],
                                360,
                                params['directionBinSize']
                                )
            info_matrix[4,2], pvalue_matrix[4,2], correlation_matrix[4,2] = info_matrix[2,4], pvalue_matrix[2,4], correlation_matrix[2,4]

        # velocity vs heading
        if data['task']=='LT' or data['task']=='legoLT' or data['task']=='legoToneLT' or data['task']=='legoSeqLT':
            info_matrix[3,4], pvalue_matrix[3,4], correlation_matrix[3,4] = assess_covariate(
                                data['velocity'],
                                data['LT_direction'],
                                data['running_ts'],
                                params['max_velocity_length'],
                                params['velocityBinSize'],
                                2,
                                1
                                )
            info_matrix[4,3], pvalue_matrix[4,3], correlation_matrix[4,3] = info_matrix[3,4], pvalue_matrix[3,4], correlation_matrix[3,4]

        elif data['task']=='OF' or data['task']=='legoOF' or data['task']=='plexiOF':
            info_matrix[3,4], pvalue_matrix[3,4], correlation_matrix[3,4] = assess_covariate(
                                data['velocity'],
                                data['heading'],
                                data['running_ts'],
                                params['max_velocity_length'],
                                params['velocityBinSize'],
                                360,
                                params['directionBinSize']
                                )
            info_matrix[4,3], pvalue_matrix[4,3], correlation_matrix[4,3] = info_matrix[3,4], pvalue_matrix[3,4], correlation_matrix[3,4]

        with h5py.File(os.path.join(working_directory,'covariates.h5'),'w') as f:
            f.create_dataset('info', data=info_matrix)
            f.create_dataset('p_value', data=pvalue_matrix)
            f.create_dataset('correlation', data=correlation_matrix)
            f.create_dataset('labels', data=labels)

# If used as standalone script
if __name__ == '__main__': 
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    data = load_data(args.session_path)
    data = preprocess_data(data, params)
    extract_covariates_session(data, params)