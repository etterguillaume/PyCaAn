#!/bin/bash
#SBATCH --account=def-wilsyl
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G      # memory; default unit is megabytes
#SBATCH --time=0-36:00           # time (DD-HH:MM)

#pytest # First verify that everything works correctly
python3 curate_dataset.py # Prepare dataset
python3 run_analysis.py --extract_tuning --extract_embedding
