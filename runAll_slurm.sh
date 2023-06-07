#!/bin/bash
#SBATCH --account=def-wilsyl
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G      # memory; default unit is megabytes
#SBATCH --time=0-36:00           # time (DD-HH:MM)

#pytest # First verify that everything works correctly
python3 PyCaAn/analysis/curate_dataset.py # Prepare dataset
python3 PyCaAn/analysis/extract_tuning_data.py # Extract tuning curves
python3 PyCaAn/analysis/extract_embedding_data.py # Extract embeddings
#python3 PyCaAn/analysis/extract_decoding.py # Extract embeddings
#python3 PyCaAn/extract_covariates.py # Extract covariates
