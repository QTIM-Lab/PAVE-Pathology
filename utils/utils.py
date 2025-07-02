import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = torch.cat([item[1] for item in batch], dim = 0)
	
	if isinstance(batch[0][2], torch.FloatTensor):
		label = torch.stack([item[2] for item in batch])
	else:
		label = torch.LongTensor([item[2] for item in batch])

	return [img, coords, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=4):
	kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None, multi_label=False, labels=None, val_frac=0.1, test_frac=0.1):
	
	if multi_label:
		from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
		
		# Create a dummy X vector, as the splitter works on data arrays
		# The labels parameter is expected to be a numpy array of shape (n_samples, n_labels)
		X = np.arange(len(labels)).reshape(len(labels), 1)
		y = np.array(labels)

		# Setup the k-fold splitter
		# n_splits is treated as the number of folds for cross-validation
		# The split will be train/test, we'll carve out a validation set from the training set
		mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

		# The generator yields train/test indices, we will use these to create train/val/test splits
		# This is a conceptual change: instead of providing val_num/test_num, we use fractions
		# of a k-fold split.
		for train_index, test_index in mskf.split(X, y):
			
			# The test set is directly from the fold
			test_ids = test_index
			
			# We need to create a validation set from the training set
			# We'll use the same stratification logic for this sub-split
			train_val_labels = y[train_index]
			X_train_val = np.arange(len(train_val_labels)).reshape(len(train_val_labels), 1)

			# Calculate the split size for validation
			# This creates a single train/validation split (n_splits=2 is not used for k-fold here)
			# A bit of a misuse of the tool, but effective. A better way might be train_test_split
			# Let's adjust to be more robust. We'll split the train set into train/val
			
			# new val frac relative to the training set size
			val_frac_adjusted = val_frac / (1.0 - test_frac)
			
			# We'll use a single split to divide the training data into a new training set and a validation set
			sub_mskf = MultilabelStratifiedKFold(n_splits=int(1/val_frac_adjusted), shuffle=True, random_state=seed)
			
			# We only need one split, so we'll break after the first iteration
			for sub_train_index, val_index in sub_mskf.split(X_train_val, train_val_labels):
				# The indices returned are relative to the `train_index` array, so we need to map them back
				val_ids = train_index[val_index]
				train_ids = train_index[sub_train_index]
				break

			yield list(train_ids), list(val_ids), list(test_ids)

	else:
		indices = np.arange(samples).astype(int)
		
		if custom_test_ids is not None:
			indices = np.setdiff1d(indices, custom_test_ids)

		np.random.seed(seed)
		for i in range(n_splits):
			all_val_ids = []
			all_test_ids = []
			sampled_train_ids = []
			
			if custom_test_ids is not None: # pre-built test split, do not need to sample
				all_test_ids.extend(custom_test_ids)

			for c in range(len(val_num)):
				possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
				val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

				remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
				all_val_ids.extend(val_ids)

				if custom_test_ids is None: # sample test split

					test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
					remaining_ids = np.setdiff1d(remaining_ids, test_ids)
					all_test_ids.extend(test_ids)

				if label_frac == 1:
					sampled_train_ids.extend(remaining_ids)
				
				else:
					sample_num  = math.ceil(len(remaining_ids) * label_frac)
					slice_ids = np.arange(sample_num)
					sampled_train_ids.extend(remaining_ids[slice_ids])

			yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	if dataset.multi_label:
		raise NotImplementedError("Weighted sampling is not supported for multi-label classification. Please disable it by removing the --weighted_sample flag.")

	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

