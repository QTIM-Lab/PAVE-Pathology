import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, num_workers=8)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, coords, label) in enumerate(loader):
        data, coords, label = data.to(device), coords.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(h=data, coords=coords)
        
        if args.n_classes == 2 and args.threshold is not None:
            Y_hat = (Y_prob[:, 1] >= args.threshold).long()
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get actual class labels from args
    if hasattr(args, 'label_dict'):
        # Create reverse mapping from numeric to string labels
        reverse_label_dict = {v: k for k, v in args.label_dict.items()}
        class_labels = [reverse_label_dict.get(i, f'Class {i}') for i in range(args.n_classes)]
    else:
        # Fallback to generic labels if no label_dict available
        class_labels = [f'Class {i}' for i in range(args.n_classes)]
    
    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Rotate x-axis labels if they're long
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save confusion matrix plot
    if hasattr(args, 'save_dir'):
        cm_save_path = os.path.join(args.save_dir, 'confusion_matrix.png')
        plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix saved to: {cm_save_path}')
    
    plt.close()
    
    # Print confusion matrix to console with actual labels
    print('\nConfusion Matrix:')
    print('True labels (rows):', class_labels)
    print('Predicted labels (columns):', class_labels)
    print(cm)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            # Annotate every other threshold point (skip the first and last for clarity)
            for i in range(1, len(fpr)-1, 2):
                plt.annotate(f'{thresholds[i]:.2f}', 
                             (fpr[i], tpr[i]), 
                             textcoords="offset points", 
                             xytext=(5,5), 
                             ha='left', fontsize=4, color='blue', rotation=0)

            if hasattr(args, 'save_dir'):
                roc_save_path = os.path.join(args.save_dir, 'roc_curve.png')
                plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
                print(f'ROC curve saved to: {roc_save_path}')
            plt.close()
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
