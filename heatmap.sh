#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
#SBATCH --gres=gpu:1
#SBATCH --mem=224GB
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --account=amc-general
#SBATCH --job-name=heatmap
#SBATCH --output="job_logs/heatmap_%J.log"
#SBATCH --error="job_logs/heatmap_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

# This script is used to create heatmaps based on a given config file.

module load miniforge

conda activate clam_latest

# Allow config file to be passed as an argument, with a default
CONFIG_FILE=${1:-normalcy_1_test.yaml}

# Run the heatmap creation script
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file "$CONFIG_FILE" --h5_files_dir /scratch/alpine/$USER/pave_training/pathology_features/h5_files