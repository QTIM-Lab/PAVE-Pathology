#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --time=12:00:00
#SBATCH --ntasks=16
#SBATCH --account=amc-general
#SBATCH --job-name=patch_pave
#SBATCH --output="job_logs/patch_%J.log"
#SBATCH --error="job_logs/patch_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

echo "Segmenting and patching for ${1}"

module load miniforge

conda activate clam_latest

python create_patches_fp.py --source "${1}/wsis" --save_dir "${1}" --patch_size 256 --preset pave_pathology.csv --seg --patch --stitch

python gen_pre_feat_ext_csv.py --input_csv "${1}/process_list_autogen.csv" --output_dir "${1}"