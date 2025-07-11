import os
from pathlib import Path
from tqdm import tqdm # For a nice progress bar (install with: pip install tqdm)

""" 
This script is used to create symlinks for all the pt, h5, and svs files that appear distributed across subdirectories (for parallel processing) in a single directory.
They have been spilt up into batches of 300 files each, lettered A-Z, for segmentation, patching, and feature extraction.
But since the training and evaluation scripts expect all the files to be in a single directory, we need to create symlinks to the files in the data_root_dir.
"""


def consolidate_files_with_symlinks(
    data_root_dir="/scratch/alpine/ataghinia@xsede.org/", 
    consolidated_folder_base="/scratch/alpine/ataghinia@xsede.org/pave_training/pathology_features/"
):
    """
    Recursively finds all specified file types within corresponding subdirectories
    and creates symbolic links to them in a new consolidated folder structure.
    """
    
    file_type_map = {
        'pt': {'source_dir': 'pt_files', 'dest_dir': 'pt_files'},
        'h5': {'source_dir': 'h5_files', 'dest_dir': 'h5_files'},
        'svs': {'source_dir': 'wsis', 'dest_dir': 'wsis'}
    }

    data_root_path = Path(data_root_dir).resolve()
    consolidated_base_path = Path(consolidated_folder_base).resolve()
    
    print(f"Starting consolidation process for: {data_root_path}")
    print(f"Target consolidated base folder: {consolidated_base_path}")

    # Create the base consolidated folder if it doesn't exist
    consolidated_base_path.mkdir(parents=True, exist_ok=True)
    
    summary = {ext: {'found': 0, 'created': 0, 'skipped_dup': 0, 'skipped_err': 0} for ext in file_type_map}

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(data_root_path):
        current_dir_path = Path(dirpath)

        for ext, mapping in file_type_map.items():
            source_dir_name = mapping['source_dir']
            
            if current_dir_path.name == source_dir_name:
                dest_dir_name = mapping['dest_dir']
                consolidated_dest_path = consolidated_base_path / dest_dir_name
                consolidated_dest_path.mkdir(exist_ok=True)
                
                print(f"\nFound source '{source_dir_name}' directory: {current_dir_path}")
                
                files_in_source = list(current_dir_path.glob(f"*.{ext}"))
                if not files_in_source:
                    print(f"  No .{ext} files found. Skipping.")
                    continue
                
                print(f"  Found {len(files_in_source)} .{ext} files to link.")
                summary[ext]['found'] += len(files_in_source)

                for src_file_path in tqdm(files_in_source, desc=f"Linking .{ext} from {current_dir_path.name}"):
                    filename = src_file_path.name
                    dst_link_path = consolidated_dest_path / filename

                    if dst_link_path.exists() or dst_link_path.is_symlink():
                        summary[ext]['skipped_dup'] += 1
                        continue

                    try:
                        dst_link_path.symlink_to(src_file_path.resolve())
                        summary[ext]['created'] += 1
                    except OSError as e:
                        print(f"  Error creating symlink for {filename}: {e}")
                        summary[ext]['skipped_err'] += 1
                    except Exception as e:
                        print(f"  An unexpected error occurred for {filename}: {e}")
                        summary[ext]['skipped_err'] += 1

    print("\n--- Consolidation Summary ---")
    for ext, counts in summary.items():
        print(f"\nFile Type: .{ext}")
        print(f"  Total files found: {counts['found']}")
        print(f"  Symbolic links created: {counts['created']}")
        print(f"  Files skipped (duplicates): {counts['skipped_dup']}")
        print(f"  Files skipped (errors): {counts['skipped_err']}")
    
    print(f"\nSuccessfully updated consolidated feature directory: {consolidated_base_path}")


# --- How to use this function ---
if __name__ == "__main__":
    consolidate_files_with_symlinks()