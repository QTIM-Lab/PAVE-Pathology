#!/bin/bash

: <<'end_comment'

This script calls the patch_template.sh script for each subdirectory in the navyblue directory.
Should be adapted for use outside Alpine (i.e. changing base dir, etc.)

end_comment

# Make sure the template is executable
chmod +x patch_template.sh

# Base directory - can be changed as needed
BASE_DIR="/scratch/alpine/$USER/navyblue"

# Loop through all subdirectories
for dir in "$BASE_DIR"/*/; do

   # Skip if not a directory
   if [[ ! -d "$dir" ]]; then
      continue
   fi
   # dir looks like /scratch/alpine/aident/navyblue/A/

   echo "Submitting job for $dir"
   sbatch --export=NONE patch_template.sh "$dir"
done 