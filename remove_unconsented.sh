#!/bin/bash

: <<'end_comment'

This script is used to delete the files in the unconsented.csv file in the target directory.
These files are the slide images and the corresponding patch, mask, and embedding files of the unconsenting patients.
They should be deleted from our servers as they come through the pipeline.

end_comment

# Set the directory and unconsented CSV to the first and second arguments, or use defaults if not provided
TARGET_DIR="${1:-/scratch/alpine/ataghinia@xsede.org/}"
UNCONSENTED_CSV="${2:-unconsented.csv}"

# Read the CSV file, skip header, get first column, remove extension
while read -r basename; do
    echo "Searching for files matching: $basename in $TARGET_DIR"
    # Find and delete files matching the basename in the target directory
    while IFS= read -r -d '' file; do
        echo "Deleting: $file"
        rm "$file"
    done < <(find "$TARGET_DIR" -name "$basename*" -type f -print0)
done < <(tail -n +2 "$UNCONSENTED_CSV" | cut -d',' -f1 | sed 's/\.[^.]*$//')