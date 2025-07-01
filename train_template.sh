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
    --no_weighted_sample)
      NO_WEIGHTED_SAMPLE="True"
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
NO_WEIGHTED_SAMPLE=${NO_WEIGHTED_SAMPLE:-False}
ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-"--early_stopping --log_data"}
USE_POS_EMBED=${USE_POS_EMBED:-False}
MULTI_LABEL=${MULTI_LABEL:-False}

echo "Training task: $TASK, code $EXP_CODE"
echo "Subtyping: $SUBTYPING, use_pos_embed: $USE_POS_EMBED"
echo "Multi-label: $MULTI_LABEL, no weighted sample: $NO_WEIGHTED_SAMPLE"

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
   $( [ "$MULTI_LABEL" = "True" ] && echo "--multi_label" ) \
   $( [ "$NO_WEIGHTED_SAMPLE" = "False" ] && echo "--weighted_sample" ) \
   $ADDITIONAL_ARGS


# source train_template.sh --task pathology_sufficiency --exp_code sufficiency_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed


# Run interactively:

# CUDA_VISIBLE_DEVICES=0 python main.py --task pathology_sufficiency --exp_code sufficiency_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --k 1 --drop_out 0.25 --lr 1e-5 --reg 1e-6 --max_epochs 100 --bag_loss ce --inst_loss svm --embed_dim 1024 --use_pos_embed --weighted_sample --early_stopping --log_data


# Job for pathology_normalcy, default hyperparameters:

# sbatch train_template.sh --task pathology_normalcy --exp_code normalcy_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed --lr 1e-4 --reg 1e-5


# Job for pathology_normalcy, tuned hyperparameters:

# sbatch train_template.sh --task pathology_normalcy --exp_code normalcy_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed


# Job for pathology_sufficiency, tuned hyperparameters:

# sbatch train_template.sh --task pathology_sufficiency --exp_code sufficiency_pos --model_type clam_sb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed


# Job for pathology_full_subtyping, tuned hyperparameters:

# sbatch train_template.sh --task pathology_full_subtyping --exp_code full_subtyping_pos --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed


# Job for pathology_sufficiency_multi_label, tuned hyperparameters:

# sbatch train_template.sh --task pathology_sufficiency_multi_label --exp_code sufficiency_multi_label_pos --model_type clam_mb --data_root_dir /scratch/alpine/$USER/pave_training --use_pos_embed --multi_label --no_weighted_sample