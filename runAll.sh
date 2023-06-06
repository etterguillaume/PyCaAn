## Run tests
#pytest # First verify that everything works correctly

## Analyze dataset
#python3 PyCaAn/analysis/curate_dataset.py # Prepare dataset
#python3 PyCaAn/analysis/extract_tuning_data.py # Extract tuning curves
#python3 PyCaAn/analysis/extract_embedding_data.py # Extract embeddings
python3 PyCaAn/analysis/extract_decoding.py # Extract embeddings
#python3 PyCaAn/extract_covariates.py # Extract covariates

#python3 PyCaAn/align_footprints.py # chronic alignment
#python3 PyCaAn/extract_aligned_neurons.py # extract aligned neurons

## Plot all results
#python3 extract_aligned_neurons.py # extract aligned neurons
