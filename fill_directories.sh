#!/bin/bash

# Base directory containing lettered folders and loose SVS files
BASE_DIR=$1
TARGET_COUNT=300

# Check if base directory is provided
if [ -z "$BASE_DIR" ]; then
    echo "Error: Please provide the base directory"
    echo "Usage: $0 /path/to/base/directory"
    exit 1
fi

# Create a temporary file to track which SVS files we've used
TEMP_FILE=$(mktemp)

# First, get a list of all loose SVS files
echo "Collecting loose SVS files..."
find "$BASE_DIR" -maxdepth 1 -name "*.svs" > "$TEMP_FILE"

# Process each lettered directory
for dir in "$BASE_DIR"/[A-Z]; do
   dirname=$(basename "$dir")
   wsis_dir="$dir/wsis"
   
   # Create wsis directory if it doesn't exist
   mkdir -p "$wsis_dir"
   
   # Count current files
   current_count=$(find "$wsis_dir" -name "*.svs" | wc -l)
   echo "Directory $dirname has $current_count files"
   
   # If we need more files
   if [ "$current_count" -lt "$TARGET_COUNT" ]; then
      needed=$((TARGET_COUNT - current_count))
      echo "Need to add $needed files to $dirname"
      
      # Move files until we reach target count or run out of files
      while [ "$needed" -gt 0 ] && [ -s "$TEMP_FILE" ]; do
         # Get the first unused SVS file
         next_file=$(head -n 1 "$TEMP_FILE")
         base_next_file=$(basename "$next_file")
         target_path="$wsis_dir/$base_next_file"

         # Check if file already exists in the target directory
         if [ -e "$target_path" ]; then
            echo "Skipping $base_next_file: already exists in $wsis_dir"
            # Remove it from our list
            sed -i '1d' "$TEMP_FILE"
            continue
         fi

         # Move it to the wsis directory
         echo "Moving $next_file to $wsis_dir"
         mv "$next_file" "$wsis_dir/"
         
         # Remove it from our list
         sed -i '1d' "$TEMP_FILE"
         
         needed=$((needed - 1))
      done
      
      if [ "$needed" -gt 0 ]; then
         echo "Warning: Not enough loose SVS files to fill $dirname"
      fi
   fi
done

# Clean up
rm "$TEMP_FILE"

echo "Done!" 