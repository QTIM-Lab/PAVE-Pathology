#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=12:00:00
#SBATCH --account=amc-general
#SBATCH --job-name=feat_ext_pave
#SBATCH --output="job_logs/feat_ext_%J.log"
#SBATCH --error="job_logs/feat_ext_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

: <<'end_comment'

This script is used to extract features from a given directory, which typically is a lettered subdirectory of 300 WSIs.

end_comment


echo "Extracting features for ${1}"

module load miniforge

conda activate clam_latest

export UNI_CKPT_PATH=checkpoints/uni/pytorch_model.bin

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir "${1}" --data_slide_dir "${1}/wsis" --csv_path "${1}/pre_feat_ext.csv" --feat_dir "${1}" --batch_size 1024 --slide_ext .svs --model_name uni_v1