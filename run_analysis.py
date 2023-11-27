import yaml
import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data

from pycaan.analysis.extract_basic_info import extract_basic_info_session
from pycaan.analysis.extract_tuning import extract_tuning_session
from pycaan.analysis.extract_embedding import extract_embedding_session
from pycaan.analysis.plot_summary import plot_summary_session
from pycaan.analysis.decode_embedding import decode_embedding_session
from pycaan.analysis.extract_covariates import extract_covariates_session
from pycaan.analysis.extract_model import extract_model_session
from pycaan.analysis.extract_model_predictions import extract_model_predictions_session

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--param_file', type=str, default='params.yaml')
    parser.add_argument('--extract_basic_info', action='store_true', default=False)
    parser.add_argument('--plot_summary', action='store_true', default=False)
    parser.add_argument('--extract_covariates', action='store_true', default=False)
    parser.add_argument('--extract_tuning', action='store_true', default=False)
    parser.add_argument('--extract_embedding', action='store_true', default=False)
    parser.add_argument('--decode_embedding', action='store_true', default=False)
    parser.add_argument('--extract_model', action='store_true', default=False)
    parser.add_argument('--fit_model', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments() #TODO add params override here
    config = vars(args)

    with open(args.param_file,'r') as file:
        params = yaml.full_load(file)

    with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
        session_file = yaml.full_load(file)
    session_list = session_file['sessions']
    with open(os.path.join(params['path_to_results'],'excludedList.yaml'),'r') as file:
        excluded_file = yaml.full_load(file)
    excluded_list = excluded_file['sessions']
    print(f'{len(session_list)} sessions to process, {len(excluded_list)} to exclude')

    for i, session in enumerate(tqdm(session_list)):
        # Load data and preprocess
        data = load_data(session)
        data = preprocess_data(data, params)

        if args.plot_summary:
            plot_summary_session(data, params)

        if args.extract_basic_info:
            extract_basic_info_session(data, params)

        if args.extract_tuning:
            extract_tuning_session(data, params)

        if args.extract_covariates:
            extract_covariates_session(data, params)

        if args.extract_model:
            extract_model_session(data, params)

        if args.fit_model:
            extract_model_predictions_session(data, params)
            
        if args.extract_embedding and session not in excluded_list: # Exclude sessions with not enough data
            extract_embedding_session(data, params)
            
        if args.decode_embedding and session not in excluded_list: # Exclude sessions with not enough data
            decode_embedding_session(data, params)