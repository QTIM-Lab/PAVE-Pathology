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
MODEL_TYPE=${MODEL_TYPE:-clam_sb}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/scratch/alpine/$USER/pave_training}
K=${K:-1}
DROP_OUT=${DROP_OUT:-0.25}
LR=${LR:-1e-4}
REG=${REG:-1e-5}
BAG_LOSS=${BAG_LOSS:-ce}
INST_LOSS=${INST_LOSS:-svm}
EMBED_DIM=${EMBED_DIM:-1024}
MAX_EPOCHS=${MAX_EPOCHS:-100}
ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-"--subtyping --weighted_sample --early_stopping --log_data"}

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
   $ADDITIONAL_ARGS
