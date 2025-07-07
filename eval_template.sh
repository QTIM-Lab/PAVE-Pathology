#!/bin/bash

#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --partition=aa100,al40
# #SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks=16
#SBATCH --time=23:59:00
#SBATCH --account=amc-general
#SBATCH --job-name=eval_template
#SBATCH --output="job_logs/eval_template_%J.log"
#SBATCH --error="job_logs/eval_template_%J.err"
#SBATCH --mail-user=aiden.taghinia@cuanschutz.edu
#SBATCH --mail-type=END

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      TASK="$2"
      shift 2
      ;;
    --save_exp_code)
      SAVE_EXP_CODE="$2"
      shift 2
      ;;
    --models_exp_code)
      MODELS_EXP_CODE="$2"
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
    --embed_dim)
      EMBED_DIM="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
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
MODELS_EXP_CODE=${MODELS_EXP_CODE:-subtyping_1} # This should be changed to the experiment code of the models you want to evaluate
SAVE_EXP_CODE=${SAVE_EXP_CODE:-"eval_${MODELS_EXP_CODE}"}
MODEL_TYPE=${MODEL_TYPE:-clam_mb}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-/scratch/alpine/$USER/pave_training}
K=${K:-1}
DROP_OUT=${DROP_OUT:-0.25}
EMBED_DIM=${EMBED_DIM:-1024}
SPLIT=${SPLIT:-test}
ADDITIONAL_ARGS=${ADDITIONAL_ARGS:-""}
THRESHOLD_ARG=${THRESHOLD:+"--threshold $THRESHOLD"}


module load miniforge

conda activate clam_latest

echo "Running evaluation with the following parameters:"
echo "Task: $TASK"
echo "Models Exp Code: $MODELS_EXP_CODE"
echo "Save Exp Code: $SAVE_EXP_CODE"
echo "Model Type: $MODEL_TYPE"
echo "Split: $SPLIT"
echo "Threshold: ${THRESHOLD:-Not Set}"
echo "Additional Args: ${ADDITIONAL_ARGS:-None}"

python eval.py \
   --task $TASK \
   --save_exp_code $SAVE_EXP_CODE \
   --models_exp_code $MODELS_EXP_CODE \
   --data_root_dir $DATA_ROOT_DIR \
   --model_type $MODEL_TYPE \
   --k $K \
   --drop_out $DROP_OUT \
   --embed_dim $EMBED_DIM \
   --split $SPLIT \
   $THRESHOLD_ARG \
   $ADDITIONAL_ARGS

# Example usage:

# source eval_template.sh --task pathology_normalcy --models_exp_code normalcy_1_s1 --save_exp_code normalcy_0.08 --model_type clam_sb --threshold 0.08

# source eval_template.sh --task pathology_normalcy --models_exp_code normalcy_1_s1 --save_exp_code normalcy_0.02 --model_type clam_sb --threshold 0.02

# source eval_template.sh --task pathology_normalcy --models_exp_code normalcy_1_s1 --save_exp_code normalcy_0.01 --model_type clam_sb --threshold 0.01


# VAL THEN TEST


# sbatch eval_template.sh --task pathology_normalcy --models_exp_code normalcy_1_s1 --save_exp_code normalcy_val --model_type clam_sb --split val

# sbatch eval_template.sh --task pathology_normalcy --models_exp_code normalcy_1_s1 --save_exp_code normalcy_test --model_type clam_sb --split test --threshold ???



# sbatch eval_template.sh --task pathology_sufficiency --models_exp_code sufficiency_1_s1 --save_exp_code sufficiency_val --model_type clam_sb --split val

# sbatch eval_template.sh --task pathology_sufficiency --models_exp_code sufficiency_1_s1 --save_exp_code sufficiency_test --model_type clam_sb --split test --threshold ???


# sbatch eval_template.sh --task pathology_sufficiency_subtyping --models_exp_code sufficiency_1_s1 --save_exp_code sufficiency_subtyping_val --model_type clam_mb --split val

# sbatch eval_template.sh --task pathology_sufficiency_subtyping --models_exp_code sufficiency_1_s1 --save_exp_code sufficiency_subtyping_test --model_type clam_mb --split test --threshold ???





