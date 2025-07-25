import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=[
    'task_1_tumor_vs_normal', 
    'task_2_tumor_subtyping', 
    'pathology_full_subtyping', 
    'pathology_sufficiency', 
    'pathology_normalcy', 
    'pathology_sufficiency_subtyping', 
    'pathology_management',
    'pathology_abnormal_subtyping'
    ])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

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

elif args.task == 'pathology_sufficiency_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_sufficiency_subtyping.csv',
                            shuffle = False,
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'sufficient':0, 'blurry':1, 'insufficient':2},
                            patient_strat=False,
                            ignore=[],)

elif args.task == 'pathology_normalcy':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_normalcy.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'abnormal':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'pathology_management':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_management.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'follow_up':0, 'treatment':1},
                            patient_strat=False,
                            ignore=[],)

elif args.task == 'pathology_abnormal_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/pathology_abnormal_subtyping.csv',
                            shuffle = False,
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'low_grade':0, 'high_grade':1, 'cancer':2},
                            patient_strat=False,
                            ignore=[],)

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

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



