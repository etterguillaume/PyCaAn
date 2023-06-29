## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset
python3 run_dataset_curation.py # Prepare dataset

#python3 run_analysis.py --extract_basic_info --extract_tuning --extract_embedding --decode_embedding

python3 run_analysis.py --extract_basic_info
#python3 run_analysis.py --extract_tuning
#python3 run_analysis.py --extract_embedding
