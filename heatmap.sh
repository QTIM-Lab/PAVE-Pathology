#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
# #SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=23:59:00
#SBATCH --account=amc-general
#SBATCH --job-name=train_template
#SBATCH --output="job_logs/train_template_%J.log"
#SBATCH --error="job_logs/train_template_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

module load miniforge

conda activate clam_latest

# Run the heatmap creation script
CUDA_VISIBLE_DEVICES=0 python PAVE-Pathology/create_heatmaps.py --config_file PAVE-Pathology/heatmaps/configs/normalcy_1_test.yaml --h5_files_dir /scratch/alpine/$USER/pave_training/pathology_features/h5_files 