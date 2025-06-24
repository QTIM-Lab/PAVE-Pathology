#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=23:59:00
#SBATCH --account=amc-general
#SBATCH --job-name=train_pave
#SBATCH --output="job_logs/train_%J.log"
#SBATCH --error="job_logs/train_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

module load miniforge

conda activate clam_latest

python create_splits_seq.py --task pathology_classifier --seed 1 --k 1

CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 1 --exp_code pathology_classifier --weighted_sample --bag_loss ce --inst_loss svm --task pathology_classifier --model_type clam_mb --log_data --subtyping --data_root_dir /scratch/alpine/$USER/pave_training --embed_dim 1024
