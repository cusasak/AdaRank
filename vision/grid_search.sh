#!/bin/bash

# This script performs grid search over specified hyperparameters for Iso_AdaRank model training.
# It iterates through combinations of learning rates and weight decays, executing the training script for each combination.


for val in 0.4 0.5 0.6 0.7
do
  echo "Running with Common_space_fraction = ${val}"
  python main.py config_list_path="configs/iso_cts_${val}.yaml"
done