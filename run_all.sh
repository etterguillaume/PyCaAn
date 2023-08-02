## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset focusing on CA1 only
#python3 run_dataset_curation.py --param_file 'params_CA1.yaml' # Prepare dataset
python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_basic_info --plot_summary
#python3 run_analysis.py --param_file 'params_CA1.yaml' --extract_tuning  --extract_embedding --extract_neural_structure --decode_embedding

## Analyze full dataset, with smaller number of neurons allowed
#python3 run_dataset_curation.py --param_file 'params_regions.yaml' # Prepare dataset
python3 run_analysis.py --param_file 'params_regions.yaml' --extract_basic_info --plot_summary
#python3 run_analysis.py --param_file 'params_regions.yaml' --extract_tuning  --extract_embedding --extract_neural_structure --decode_embedding --align_embeddings
