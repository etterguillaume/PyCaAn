## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset
python3 run_dataset_curation.py --param_file 'params_regions.yaml' # Prepare dataset
python3 run_analysis.py --param_file 'params_regions.yaml' --extract_basic_info --plot_summary
python3 run_analysis.py --param_file 'params_regions.yaml' --extract_tuning  --extract_embedding --extract_neural_structure --decode_embedding --align_embeddings
