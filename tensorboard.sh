#!/bin/bash

# This script is used to start a tensorboard server in the PAVE-Pathology repo.
# Defaults to results/ directory.

DIR=${1:-results/}

module load miniforge

conda activate clam_latest

tensorboard --logdir $DIR