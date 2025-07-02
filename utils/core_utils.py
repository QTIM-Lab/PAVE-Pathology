import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import pandas as pd

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = Y_hat.cpu().numpy()
        Y = Y.cpu().numpy()

        if Y.ndim == 1: # In single-label classification, we f
            Y_hat = Y_hat.flatten()
            for i in range(self.n_classes):
                mask = (Y == i)
                self.data[i]["count"] += np.sum(mask)
                self.data[i]["correct"] += np.sum(Y_hat[mask] == i)
        else:
            for i in range(self.n_classes):
                self.data[i]["count"] += Y.shape[0]
                self.data[i]["correct"] += np.equal(Y_hat[:, i], Y[:, i]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.multi_label:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.multi_label:
            model_dict.update({'multi_label': True})
        
        model_dict.update({'use_pos_embed': args.use_pos_embed})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, 
                                       stop_epoch=30, # Lower from 50 to 30, because loss is not improving
                                       verbose = True)

    else:
        early_stopping = None
    print('Done!')

    checkpoint_path = os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))
    
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, args.multi_label)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.multi_label)
        
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.multi_label)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.multi_label)
        
        # Save checkpoint every N epochs (override previous)
        """ if epoch % 10 == 0:  # Every 10 epochs
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint at epoch {epoch}') """
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _= summary(model, val_loader, args.n_classes, args.multi_label, args.model_type)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes, args.multi_label, args.model_type)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, multi_label=False):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    successful_batches = 0

    print('\n')
    for batch_idx, (data, coords, label) in enumerate(loader):
        try:
            data, coords, label = data.to(device), coords.to(device), label.to(device)
            if multi_label:
                label = label.float()
            
            logits, Y_prob, Y_hat, _, instance_dict = model(h=data, coords=coords, label=label, instance_eval=True)
            
            acc_logger.log_batch(Y_hat, label)
            loss_bag = loss_fn(logits, label)
            loss_value = loss_bag.item()

            if not multi_label:
                instance_loss = instance_dict['instance_loss']
                inst_count+=1
                instance_loss_value = instance_loss.item()
                train_inst_loss += instance_loss_value
                total_loss = bag_weight * loss_bag + (1-bag_weight) * instance_loss 
                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)
            else:
                total_loss = loss_bag
                instance_loss_value = 0.0

            train_loss += loss_value
            successful_batches += 1
            
            if (batch_idx + 1) % 20 == 0:
                print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                    'label: {}, bag_size: {}'.format(label.squeeze(0), data.size(0)))

            error = 1. - Y_hat.eq(label).float().mean().item()
            train_error += error
            
            # backward pass
            total_loss.backward()
            # step
            optimizer.step()
            optimizer.zero_grad()
            
        except Exception as e:
            print(f'Error in batch {batch_idx}: {e}')
            continue  # Skip this batch and continue training

    # calculate loss and error for epoch (only successful batches)
    if successful_batches > 0:
        train_loss /= successful_batches
        train_error /= successful_batches
    else:
        print("Warning: No successful batches in this epoch!")
        return
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, multi_label=False):   
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        if multi_label:
            label = label.float()

        logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log_batch(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.squeeze(0), data.size(0)))
           
        error = 1. - Y_hat.eq(label).float().mean().item()
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None, multi_label=False):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            if multi_label:
                label = label.float()

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log_batch(Y_hat, label)
            
            loss = loss_fn(logits, label)
            
            val_loss += loss.item()
            error = 1. - Y_hat.eq(label).float().mean().item()
            val_error += error
            
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    val_error /= len(loader)
    val_loss /= len(loader)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if n_classes == 2 and not multi_label:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        if multi_label:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            auc = np.nanmean(np.array(aucs))
    
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None, multi_label=False):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    all_probs = []
    all_labels = []

    with torch.inference_mode():
        for batch_idx, (data, coords, label) in enumerate(loader):
            data, coords, label = data.to(device), coords.to(device), label.to(device)
            if multi_label:
                label = label.float()

            with torch.no_grad():
                logits, Y_prob, Y_hat, _, instance_dict = model(h=data, coords=coords, label=label, instance_eval=True)

            acc_logger.log_batch(Y_hat, label)
            
            loss_bag = loss_fn(logits, label)
            val_loss += loss_bag.item()

            if not multi_label:
                instance_loss = instance_dict['instance_loss']
                inst_count+=1
                instance_loss_value = instance_loss.item()
                val_inst_loss += instance_loss_value
                inst_preds = instance_dict['inst_preds']
                inst_labels = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds, inst_labels)
            
            error = 1. - Y_hat.eq(label).float().mean().item()
            val_error += error

            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    val_error /= len(loader)
    val_loss /= len(loader)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if n_classes == 2 and not multi_label:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        if multi_label:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, n_classes, multi_label=False, model_type='clam_sb'):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    patient_results = {}

    all_probs = []
    all_labels = []
    
    for batch_idx, batch in enumerate(loader):
        data, label = batch[0], batch[1]
        data, label = data.to(device), label.to(device)
        
        slide_id = loader.dataset.slide_data['slide_id'][batch_idx]
        
        if multi_label:
            label = label.float()

        with torch.no_grad():
            if 'clam' in model_type:
                coords = batch[2].to(device)
                logits, Y_prob, Y_hat, _, _ = model(h=data, coords=coords)
            else:
                logits, Y_prob, Y_hat, _, _ = model(data)
        
        acc_logger.log_batch(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(label.cpu().numpy())
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.cpu().numpy()}})

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if multi_label:
        auc = roc_auc_score(all_labels, all_probs, average='macro')
        test_error = 1. - np.mean([np.all(all_labels[i] == (all_probs[i] > 0.5)) for i in range(len(all_labels))])
    else:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            aucs = []
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            auc = np.nanmean(np.array(aucs))

        all_preds = np.argmax(all_probs, axis=1)
        test_error = 1.0 - np.sum(all_preds == all_labels) / len(all_labels)


    return patient_results, test_error, auc, acc_logger
