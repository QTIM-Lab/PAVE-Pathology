import json
import os
import sys

TASK_PATH = 'tasks.json'

try:
    with open(TASK_PATH, 'r') as f:
        tasks = json.load(f)
except FileNotFoundError:
    print(f"Error: Tasks configuration file not found at {TASK_PATH}")
    sys.exit(1)

def get_dataset_args(args, task_name):
    """
    Updates args with task-specific configurations and returns the dataset arguments.
    """
    
    task_config = tasks.get(task_name)
    if task_config is None:
        raise ValueError(f"Task '{task_name}' not found in {TASK_PATH}.")
    
    # Update args with task-specified parameters
    for key, value in task_config.items():
        if key not in ['data_dir_suffix', 'csv_path', 'label_dict', 'patient_strat', 'ignore']:
            
            #print(f"Task config: Setting args.{key} = {value}")
            setattr(args, key, value)

    dataset_args = {
        'csv_path': task_config['csv_path'],
        'shuffle': task_config['shuffle'],
        'seed': args.seed,
        'print_info': task_config['print_info'],
        'label_dict': task_config['label_dict'],
        'patient_strat': task_config['patient_strat'],
        'ignore': task_config['ignore']
    }

    # Handle data directory
    if hasattr(args, 'data_root_dir'):
        # If there's no root dir in the args, dataset constructor does not need a data dir
        if 'data_dir_suffix' in task_config:
            dataset_args['data_dir'] = os.path.join(args.data_root_dir, task_config['data_dir_suffix'])
        else:
            dataset_args['data_dir'] = args.data_root_dir
        
    return dataset_args
