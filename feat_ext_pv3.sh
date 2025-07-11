#!/bin/bash

: <<'end_comment'

This script calls the feat_ext_template.sh script for each subdirectory in the pv3 directory.
Should be adapted for use outside Alpine (i.e. changing base dir, etc.)

end_comment

# Make sure the template is executable
chmod +x feat_ext_template.sh

# Base directory - can be changed as needed
BASE_DIR="/scratch/alpine/$USER/pv3"

# Loop through all subdirectories
for dir in "$BASE_DIR"/*/; do

   # Skip if not a directory
   if [[ ! -d "$dir" ]]; then
      continue
   fi
   # dir looks like /scratch/alpine/aident/navyblue/A/

   echo "Submitting job for $dir"
   sbatch --export=NONE feat_ext_template.sh "$dir"
done 