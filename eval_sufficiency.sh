#!/bin/bash

module load miniforge

conda activate clam_latest

CUDA_VISIBLE_DEVICES=0 python eval.py \
   --models_exp_code sufficiency_1_s1 \
   --save_exp_code sufficiency \
   --task pathology_sufficiency \
   --k 1 \
   --drop_out 0.25 \
   --model_type clam_sb \
   --results_dir results \
   --data_root_dir /scratch/alpine/$USER/pave_training \
   --embed_dim 1024
