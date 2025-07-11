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

: <<'end_comment'

This script is used to train a model on a given task with a given set of hyperparameters.
The arguments to this script are the same as the arguments to main.py itself, i.e., one could replace `sbatch train_template.sh` with `python main.py` 
(with a few desired defaults, e.g. --weighted_sample --early_stopping --log_data)

Sample uses are given below:

sbatch train_template.sh --task pathology_normalcy --exp_code normalcy_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed --lr 1e-4 --reg 1e-5
sbatch train_template.sh --task pathology_normalcy --exp_code normalcy_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed

sbatch train_template.sh --task pathology_sufficiency --exp_code sufficiency_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed

sbatch train_template.sh --task pathology_full_subtyping --exp_code full_subtyping_pos --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed

sbatch train_template.sh --task pathology_sufficiency_subtyping --exp_code sufficiency_subtyping_pos --model_type clam_mb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed
sbatch train_template.sh --task pathology_sufficiency_subtyping --exp_code sufficiency_subtyping --model_type clam_mb --data_root_dir /scratch/alpine/$USER/pave_training

sbatch train_template.sh --task pathology_management --exp_code management --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training

end_comment

# Parse command line arguments
echo "Parsing command line arguments"
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      TASK="$2"
      shift 2
      ;;
    --exp_code)
      EXP_CODE="$2"
      shift 2
      ;;
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --data_root_dir)
      DATA_ROOT_DIR="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --drop_out)
      DROP_OUT="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --reg)
      REG="$2"
      shift 2
      ;;
    --bag_loss)
      BAG_LOSS="$2"
      shift 2
      ;;
    --inst_loss)
      INST_LOSS="$2"
      shift 2
      ;;
    --embed_dim)
      EMBED_DIM="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --subtyping)
      SUBTYPING="True"
      shift 1
      ;;
    --multi_label)
      MULTI_LABEL="True"
      shift 1
      ;;
    --use_pos_embed)
      USE_POS_EMBED="True"
      shift 1
      ;;
    --additional_args)
      ADDITIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Set defaults for any unset parameters
TASK=${TASK:-pathology_full_subtyping}
EXP_CODE=${EXP_CODE:-subtyping_1}
MODEL_TYPE=${MODEL_TYPE:-clam_mb}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/scratch/alpine/$USER/pave_training}
K=${K:-1}
DROP_OUT=${DROP_OUT:-0.25}
LR=${LR:-1e-5}
REG=${REG:-1e-6}
BAG_LOSS=${BAG_LOSS:-ce}
INST_LOSS=${INST_LOSS:-svm}
EMBED_DIM=${EMBED_DIM:-1024}
MAX_EPOCHS=${MAX_EPOCHS:-100}
SUBTYPING=${SUBTYPING:-False}
ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-"--weighted_sample --early_stopping --log_data"}
USE_POS_EMBED=${USE_POS_EMBED:-False}

echo "Training task: $TASK, code $EXP_CODE"
echo "Subtyping: $SUBTYPING, use_pos_embed: $USE_POS_EMBED"

module load miniforge

conda activate clam_latest

python create_splits_seq.py --task $TASK --seed 1 --k $K

CUDA_VISIBLE_DEVICES=0 python main.py \
   --task $TASK \
   --exp_code $EXP_CODE \
   --data_root_dir $DATA_ROOT_DIR \
   --model_type $MODEL_TYPE \
   --k $K \
   --drop_out $DROP_OUT \
   --lr $LR \
   --reg $REG \
   --max_epochs $MAX_EPOCHS \
   --bag_loss $BAG_LOSS \
   --inst_loss $INST_LOSS \
   --embed_dim $EMBED_DIM \
   $( [ "$SUBTYPING" = "True" ] && echo "--subtyping" ) \
   $( [ "$USE_POS_EMBED" = "True" ] && echo "--use_pos_embed" ) \
   $ADDITIONAL_ARGS
