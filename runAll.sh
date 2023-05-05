## Run tests
python3 run_tests.py # First verify that everything works correctly

## Analyze dataset
python3 curate_dataset.py # Prepare dataset
python3 extract_tuning_data.py # Extract tuning curves
python3 extract_embedding_data.py # Extract embeddings
python3 extract_covariates.py # Extract covariates
#python3 align_footprints.py # chronic alignment
#python3 extract_aligned_neurons.py # extract aligned neurons

## Plot all results
#python3 extract_aligned_neurons.py # extract aligned neurons
