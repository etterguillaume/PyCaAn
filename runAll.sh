## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset
#python3 analysis/curate_dataset.py # Prepare dataset
#python3 analysis/extract_tuning_data.py # Extract tuning curves
python3 analysis/extract_embedding_data.py # Extract embeddings
#python3 extract_covariates.py # Extract covariates
#python3 align_footprints.py # chronic alignment
#python3 extract_aligned_neurons.py # extract aligned neurons

## Plot all results
#python3 extract_aligned_neurons.py # extract aligned neurons
