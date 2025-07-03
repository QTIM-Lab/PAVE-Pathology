#!/bin/bash

DIR=${1:-results/}

module load miniforge

conda activate clam_latest

tensorboard --logdir $DIR