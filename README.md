## Readme


# Installation and usage
To use internal functions in other projects:
`
pip install -e .
`

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

