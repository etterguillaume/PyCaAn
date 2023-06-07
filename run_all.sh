## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset
python3 curate_dataset.py # Prepare dataset
python3 run_analysis.py --extract_tuning --extract_embedding
