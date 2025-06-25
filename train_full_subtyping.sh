#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
# #SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=23:59:00
#SBATCH --account=amc-general
#SBATCH --job-name=train_full
#SBATCH --output="job_logs/train_full_subtyping_%J.log"
#SBATCH --error="job_logs/train_full_subtyping_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

module load miniforge

conda activate clam_latest

python create_splits_seq.py --task pathology_full_subtyping --seed 1 --k 1

CUDA_VISIBLE_DEVICES=0 python main.py \
   --task pathology_full_subtyping \
   --exp_code subtyping_1 \
   --data_root_dir /scratch/alpine/$USER/pave_training \
   --model_type clam_mb \
   --subtyping \
   --weighted_sample \
   --early_stopping \
   --log_data \
   --k 1 \
   --drop_out 0.25 \
   --lr 1e-4 \
   --reg 1e-5 \
   --bag_loss ce \
   --inst_loss svm \
   --embed_dim 1024 \
