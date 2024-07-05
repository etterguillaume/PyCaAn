## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset focusing on CA1 only
#python3 run_dataset_curation.py --param_file 'params_CA1.yaml' # Prepare dataset
#python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_basic_info
#python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_tuning
#python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_embedding
#python3 run_analysis.py --param_file 'params_CA1.yaml' --decode_embedding
python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_internal_tuning
# python3 run_analysis.py --param_file 'params_CA1.yaml' --fit_model
#python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_basic_info --extract_tuning --extract_embedding --decode_embedding --extract_internal_tuning
