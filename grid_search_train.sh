#!/bin/bash

# Grid search script for hyperparameter tuning
# Iterates through combinations of lr, weight decay (reg), and dropout values

# Define the hyperparameter values to search over
LR_VALUES=("1e-5" "1e-4" "1e-3")
REG_VALUES=("1e-6" "1e-5" "1e-4")
DROP_OUT_VALUES=("0.25")

# Counter for experiment codes
exp_counter=1

echo "Starting grid search with the following hyperparameter combinations:"
echo "Learning rates: ${LR_VALUES[@]}"
echo "Weight decay (reg): ${REG_VALUES[@]}"
echo "Dropout: ${DROP_OUT_VALUES[@]}"
echo "Total combinations: $(( ${#LR_VALUES[@]} * ${#REG_VALUES[@]} * ${#DROP_OUT_VALUES[@]} ))"
echo ""

# Iterate through all combinations
for lr in "${LR_VALUES[@]}"; do
    for reg in "${REG_VALUES[@]}"; do
        for dropout in "${DROP_OUT_VALUES[@]}"; do
            # Create experiment code
            exp_code="grid_search_${exp_counter}_lr${lr}_reg${reg}_drop${dropout}"
            
            echo "Submitting job for combination ${exp_counter}:"
            echo "  LR: $lr, Reg: $reg, Dropout: $dropout"
            echo "  Exp code: $exp_code"
            
            # Submit the job using sbatch
            sbatch --export=NONE train_template.sh \
                --exp_code "$exp_code" \
                --drop_out "$dropout" \
                --lr "$lr" \
                --reg "$reg" \
            
            echo "Job submitted successfully!"
            echo ""
            
            # Increment counter
            ((exp_counter++))
        done
    done
done

echo "Grid search complete! Submitted $((exp_counter - 1)) jobs."