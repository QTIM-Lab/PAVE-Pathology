import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np
import math

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'pathology_full_subtyping', 'pathology_sufficiency', 'pathology_normalcy', 'pathology_sufficiency_multi_label'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--splits_dir', type=str, default='splits',
                    help='directory to save splits (default: splits)')
parser.add_argument('--k_fold', action='store_true',
                    help='use k-fold cross-validation')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'pathology_full_subtyping':
    args.n_classes=5 #6
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_full_subtyping.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'insufficient':0, 'normal':1, 'low_grade':2, 'high_grade':3, 'cancer':4},# 'atypia':5},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'pathology_sufficiency':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_sufficiency.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'insufficient':0, 'sufficient':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'pathology_sufficiency_multi_label':
    args.n_classes=6
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_sufficiency.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_cols = ['insufficient', 'scant_material', 'blurry', 'mucus', 'scant_cells', 'inflammation'],
                            patient_strat=False,
                            ignore=[])

elif args.task == 'pathology_normalcy':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_normalcy.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'abnormal':1},
                            patient_strat=False,
                            ignore=[])
else:
    raise NotImplementedError

test_num = math.ceil(len(dataset) * args.test_frac)

if dataset.multi_label:
    val_num = args.val_frac
    test_num = args.test_frac
elif args.task == 'pathology_full_subtyping':
    val_num = np.round(len(dataset) * args.val_frac).astype(int)
    test_num = np.round(len(dataset) * args.test_frac).astype(int)

if __name__ == '__main__':
    import shutil

    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/' + str(args.task) + '_{}'.format(int(lf * 100))
        # Clear out the split_dir if it exists, then recreate it
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))

    if args.k_fold:
        for i in range(args.k):
            print(f"Creating fold {i+1}/{args.k}")
            split_dir = os.path.join(args.splits_dir, f'fold_{i}')
            os.makedirs(split_dir, exist_ok=True)
            
            dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
    else:
        dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num, val_frac=args.val_frac, test_frac=args.test_frac)
        splits = dataset.return_splits(from_id=True)
        save_splits(splits, ['train', 'val', 'test'], os.path.join(args.splits_dir, 'splits_0.csv'))

    print("Splits created successfully")
    print(f"Find splits in {args.splits_dir}")
    
    descriptor = dataset.test_split_gen(return_descriptor=True)



