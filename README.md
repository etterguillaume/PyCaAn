<p align="center">
  <img width="60%" src="docs/logo.png">
</p>

PyCaAn stands for  **Python Calcium imaging Analysis** (pronounced 'puh-can')
This repository contains tools to analyze large datasets of calcium imaging data, plot, and extract statistics.

Features:
- Tuning curves
- Unbiased information theoretic metrics
- Low-dimensional embeddings (manifolds)
- Data simulator
- Dataframe support for quick plotting/stats
- and more!

# Installation and usage
To use internal functions in other projects:
`
pip install -e .
`

Then any function can be called using:
`
import PyCaAn
`

Example: binarize calcium transients:
`
binarized_traces, neural_data = PyCaAn.binarize_ca_traces(ca_traces, z_threshold, sampling_frequency)
`


# Batch processing
First define your input dataset and output results paths in params.yaml
To reproduce all figures/results, run:
`
sh runAll.sh
`

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

# Configuration

