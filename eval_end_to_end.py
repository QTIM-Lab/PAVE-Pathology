from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='/scratch/alpine/ataghinia@xsede.org/pave_training',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default='end_to_end',
                    help='experiment code to save eval results')
parser.add_argument('--test_csv', type=str, default="e2e_test.csv")

parser.add_argument('--suff_model_exp_code', type=str, default='e2e_sufficiency_s1')
parser.add_argument('--norm_model_exp_code', type=str, default='e2e_normalcy_s1')
parser.add_argument('--priority_model_exp_code', type=str, default='e2e_priority_s1')

parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))

args.model_type = "clam_sb"
args.model_size = "small"

suff_model_ckpt = os.path.join(args.results_dir, str(args.suff_model_exp_code), "s_0_checkpoint.pt")
norm_model_ckpt = os.path.join(args.results_dir, str(args.norm_model_exp_code), "s_0_checkpoint.pt")
priority_model_ckpt = os.path.join(args.results_dir, str(args.priority_model_exp_code), "s_0_checkpoint.pt")

os.makedirs(args.save_dir, exist_ok=True)

args.label_dict = {'insufficient': 0, 'normal': 1, 'low_grade': 2, 'high_grade': 3, 'cancer': 4}

def gen_dataset(df):
    return Generic_MIL_Dataset(
        slide_data_df=df,
        data_dir=os.path.join(args.data_root_dir, 'pathology_features'),
        shuffle=False,
        print_info=True,
        label_dict=args.label_dict,
        patient_strat=False,
        ignore=[]
    )

def inference(df, col, args, model_ckpt, dataset, threshold=None):
    model = initiate_model(args, model_ckpt)
    model.eval()

    loader = get_simple_loader(dataset, num_workers=8)
    slide_ids = loader.dataset.slide_data['slide_id']

    for batch_idx, (data, coords, label) in enumerate(loader):
        slide_id = slide_ids.iloc[batch_idx]
        data, coords, label = data.to(device), coords.to(device), label.to(device)

        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(h=data, coords=coords)

            if threshold is not None:
                # For binary classification, use the probability of class 1
                Y_hat = (Y_prob[:, 1] >= threshold).long()

        # Find the index in the dataframe corresponding to this slide_id
        # If slide_id is the index, use .loc, else use .index
        if slide_id in df.index:
            df.at[slide_id, col] = Y_hat.item()
        else:
            # Try to find the row with this slide_id in a column named 'slide_id'
            if 'slide_id' in df.columns:
                idx = df.index[df['slide_id'] == slide_id]
                if len(idx) > 0:
                    df.at[idx[0], col] = Y_hat.item()
                else:
                    print(f"Warning: slide_id {slide_id} not found in DataFrame.")
            else:
                print(f"Warning: slide_id {slide_id} not found in DataFrame.")

if __name__ == "__main__":
    df = pd.read_csv(os.path.join('dataset_csv', args.test_csv))

    # Ensure slide_id is the index for easier assignment
    if 'slide_id' in df.columns:
        df.set_index('slide_id', inplace=True)

    df['suff_pred'] = pd.NA
    df['norm_pred'] = pd.NA
    df['priority_pred'] = pd.NA

    # Sufficiency Model
    dataset = gen_dataset(df)
    args.n_classes = 2
    inference(df, 'suff_pred', args, suff_model_ckpt, dataset, threshold=0.8)

    # Normalcy Model
    # Only keep slides predicted as sufficient (suff_pred == 1)
    norm_df = df[df['suff_pred'] == 1].copy()
    if len(norm_df) > 0:
        norm_dataset = gen_dataset(norm_df)
        args.n_classes = 2
        inference(df, 'norm_pred', args, norm_model_ckpt, norm_dataset, threshold=0.06)
    else:
        print("No slides predicted as sufficient for normalcy model.")

    # Priority Model
    # Only keep slides predicted as abnormal (norm_pred == 0)
    priority_df = df[df['norm_pred'] == 0].copy()
    if len(priority_df) > 0:
        priority_dataset = gen_dataset(priority_df)
        args.model_type = 'clam_mb'
        args.n_classes = 3
        inference(df, 'priority_pred', args, priority_model_ckpt, priority_dataset)
    else:
        print("No slides predicted as abnormal for priority model.")

    # Reset index if needed and save results
    df.reset_index(inplace=True)
    df.to_csv(os.path.join(args.save_dir, "results.csv"), index=False)

'''
module load miniforge
conda activate clam_latest
CUDA_VISIBLE_DEVICES=0 python eval_end_to_end.py
'''
