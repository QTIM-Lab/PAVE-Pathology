#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval.py \
   --models_exp_code pathology_classifier \
   --save_exp_code pathology_classifier \
   --task pathology_classifier \
   --k 1 \
   --drop_out 0.25 \
   --model_type clam_mb \
   --results_dir results \
   --data_root_dir /scratch/alpine/$USER/pave_training \
   --embed_dim 1024
