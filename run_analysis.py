import yaml
import numpy as np
import os
from argparse import ArgumentParser
from tqdm import tqdm
from pycaan.functions.dataloaders import load_data
from pycaan.functions.signal_processing import preprocess_data
from pycaan.analysis.extract_tuning import extract_tuning_session
from pycaan.analysis.extract_embedding import extract_embedding_session
from pycaan.analysis.extract_basic_info import extract_basic_info_session

def get_arguments(): #TODO add params override here
    parser = ArgumentParser()
    parser.add_argument('--extract_basic_info', action='store_true', default=False)
    parser.add_argument('--extract_tuning', action='store_true', default=False)
    parser.add_argument('--extract_embedding', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    config = vars(args)

    with open('params.yaml','r') as file:
        params = yaml.full_load(file)

    with open(os.path.join(params['path_to_results'],'sessionList.yaml'),'r') as file:
        session_file = yaml.full_load(file)
    session_list = session_file['sessions']
    print(f'{len(session_list)} sessions to process')

    for i, session in enumerate(tqdm(session_list)):
        # Load data and preprocess
        data = load_data(session)
        data = preprocess_data(data, params)

        if args.extract_basic_info:
            extract_basic_info_session(data, params)
        if args.extract_tuning:
            extract_tuning_session(data, params)
        if args.extract_embedding:
            extract_embedding_session(data, params)

