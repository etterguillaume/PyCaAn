<p align="center">
  <img width="60%" src="docs/logo.png">
</p>

PyCaAn stands for  **Python Calcium imaging Analysis**
This repository contains tools to analyze large datasets of calcium imaging data, plot, and extract statistics.

Features:
- Tuning curves
- Unbiased information theoretic metrics
- Low-dimensional embeddings (manifolds)
- Data simulator
- Dataframe support for quick plotting/stats
- and more!

# Installation
To use internal functions in other projects, you need to install PyCaAn in your preferred environment.
Currently, this is done with a developer install:
`
pip install -e .
`

# Run analyses
PyCaAn provides both high- and low-level access to analytical functions.

## Low level
Any function can be called using:
`
import pycaan
`

Example: to binarize calcium transients, use:
`
from pycaan import binarize_ca_traces
binarized_traces, neural_data = binarize_ca_traces(ca_traces, z_threshold, sampling_frequency)
`

## High level
First define your paths (input dataset, output result folders) and other parameters in params.yaml,
you can analyze a single session from the terminal using:
`
% python3 pycaan//analysis/extract_tuning_data.py --session_path ../../example_dataset/example_region/example_subject/example_mouse
`

you can also perform a specific type of analysis on selected sessions (batch process) after running run_dataset_curation.py:
`
% python3 run_analysis.py --extract_tuning
`

Finally, for a fully automatized dataset analysis, you can run:
`
sh runAll.sh
`
This function will curate your dataset with desired threshold (e.g. minimum number of neurons per recording) and run all analyses.

# Dataset naming convention
Dataset path has to be specified in params.yaml
The naming convention should follow these principles:
`
region/subject/subject_task_condition1_condition2_..._date
`
For example:
`
amygdala/F173/F173_OF_darkness_20230804
`
