import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from itertools import cycle

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T

def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 200):
    """
    Fit a single temperature by minimizing cross-entropy on provided logits/labels.
    - val_logits: [N, C]
    - val_labels: [N]
    Returns the learned scalar temperature (float) and a TemperatureScaler module.
    """
    scaler = TemperatureScaler().to(val_logits.device)
    criterion = nn.CrossEntropyLoss()

    # LBFGS often converges quickly for a single parameter
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(val_logits)
        loss = criterion(scaled, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    # Light Adam refinement
    adam = optim.Adam(scaler.parameters(), lr=1e-2)
    for _ in range(max_iter):
        adam.zero_grad()
        loss = criterion(scaler(val_logits), val_labels)
        loss.backward()
        adam.step()

    T = torch.exp(scaler.log_T).item()
    return T, scaler

def plot_multiclass_roc(all_labels, all_probs, n_classes, save_dir, class_labels):
    all_labels_b = label_binarize(all_labels, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_b[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_b.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_labels[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    save_path = os.path.join(save_dir, 'multiclass_roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-class ROC curve saved to {save_path}")

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

    # First pass: collect logits, labels, and slide_ids
    slide_ids = loader.dataset.slide_data['slide_id']
    logits_list = []
    labels_list = []
    slide_id_list = []
    for batch_idx, (data, coords, label) in enumerate(loader):
        data, coords, label = data.to(device), coords.to(device), label.to(device)
        with torch.no_grad():
            logits, _, _, _, _ = model(h=data, coords=coords)
        logits_list.append(logits.detach().cpu())
        labels_list.append(label.detach().cpu())
        slide_id_list.append(slide_ids.iloc[batch_idx])

    # Stack collected tensors
    all_logits = torch.cat(logits_list, dim=0)
    all_tlabels = torch.cat(labels_list, dim=0).long()

    # Apply class-specific probability adjustment to logits if provided
    if args.n_classes > 2 and hasattr(args, 'prob_adjust') and args.prob_adjust is not None:
        adjust = torch.tensor(args.prob_adjust, device=all_logits.device, dtype=all_logits.dtype)
        logits_for_calib = all_logits - adjust
    else:
        logits_for_calib = all_logits

    # Temperature handling: either optimize or use provided
    T = 1.0
    if hasattr(args, 'temperature_optimize') and args.temperature_optimize:
        T, _ = fit_temperature(logits_for_calib, all_tlabels)
        print(f"Fitted temperature: {T:.4f}")
    elif hasattr(args, 'temperature') and args.temperature is not None:
        T = float(args.temperature)
        print(f"Using provided temperature: {T:.4f}")

    # Compute probabilities and predictions using temperature
    scaled_logits = logits_for_calib / T
    Y_prob_all = torch.softmax(scaled_logits, dim=1)

    if args.n_classes == 2 and hasattr(args, 'threshold') and args.threshold is not None:
        Y_hat_all = (Y_prob_all[:, 1] >= args.threshold).long()
    elif hasattr(args, 'thresholds') and args.thresholds is not None and len(args.thresholds) == args.n_classes:
        thresholds = torch.tensor(args.thresholds, device=Y_prob_all.device)
        Y_hat_all = (Y_prob_all >= thresholds).long().argmax(dim=1)
    else:
        Y_hat_all = Y_prob_all.argmax(dim=1)

    # Build outputs and metrics
    all_probs = Y_prob_all.numpy()
    all_labels = all_tlabels.numpy()
    all_preds = Y_hat_all.numpy()

    # Accuracy logger and error
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    total_error = 0.0
    for i in range(len(all_labels)):
        y_hat_tensor = torch.tensor(all_preds[i]).view(1)
        y_tensor = torch.tensor(all_labels[i]).view(1)
        acc_logger.log(y_hat_tensor, y_tensor)
        total_error += calculate_error(y_hat_tensor, y_tensor)

    test_error = total_error / len(all_labels)

    # patient_results
    patient_results = {}
    for i, slide_id in enumerate(slide_id_list):
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': all_probs[i], 'label': int(all_labels[i])}})

    # Save intermediate results if requested
    if hasattr(args, 'save_intermediate_results') and args.save_intermediate_results:
        results_dict_inter = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict_inter.update({'p_{}'.format(c): all_probs[:,c]})
        df_inter = pd.DataFrame(results_dict_inter)
        df_inter.to_csv(os.path.join(args.save_dir, 'intermediate_results.csv'), index=False)

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
            for i in range(1, len(fpr)-1):
                plt.annotate(f'{thresholds[i]:.2f}', 
                             (fpr[i], tpr[i]), 
                             textcoords="offset points", 
                             xytext=(0,0), 
                             ha='left', fontsize=4, color='black', rotation=0)

            if hasattr(args, 'save_dir'):
                roc_save_path = os.path.join(args.save_dir, 'roc_curve.png')
                plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
                print(f'ROC curve saved to: {roc_save_path}')
            plt.close()
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            plot_multiclass_roc(all_labels, all_probs, args.n_classes, args.save_dir, class_labels)
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
