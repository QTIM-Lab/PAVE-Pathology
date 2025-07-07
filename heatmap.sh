#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --ntasks=16
#SBATCH --time=23:59:00
#SBATCH --account=amc-general
#SBATCH --job-name=heatmap
#SBATCH --output="job_logs/heatmap_%J.log"
#SBATCH --error="job_logs/heatmap_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

module load miniforge

conda activate clam_latest

# Run the heatmap creation script
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config_file normalcy_1_test.yaml --h5_files_dir /scratch/alpine/$USER/pave_training/pathology_features/h5_files