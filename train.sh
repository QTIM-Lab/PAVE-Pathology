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

python create_splits_seq.py --task pathology_classifier --seed 1 --k 4

CUDA_VISIBLE_DEVICES=0 python main.py \
   --task pathology_classifier \
   --exp_code pathology_classifier \
   --data_root_dir /scratch/alpine/$USER/pave_training \
   --model_type clam_mb \
   --subtyping \
   --weighted_sample \
   --early_stopping \
   --log_data \
   --k 4 \
   --drop_out 0.5 \
   --lr 2e-4 \
   --reg 2e-4 \
   --bag_loss ce \
   --inst_loss svm \
   --embed_dim 1024 \
