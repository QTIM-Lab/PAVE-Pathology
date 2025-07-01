import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)
	print()

class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		label_cols = [],
		patient_voting = 'max',
		use_h5 = False,
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			label_col (str): Name of the column to use as the label for single-label classification
			label_cols (list): List of column names to use as labels for multi-label classification
			ignore (list): List containing class labels to ignore
			use_h5 (boolean): Whether to use h5 files
		"""
		self.use_h5 = use_h5
		self.label_cols = label_cols
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids = (None, None, None)
		self.data_dir = None

		if len(self.label_cols) > 0:
			self.multi_label = True
			self.num_classes = len(self.label_cols)
			self.label_dict = {col: i for i, col in enumerate(self.label_cols)} # {label_col: i}, no need to pass in label_dict for multi-label classification
		else:
			self.multi_label = False
			self.label_dict = label_dict
			self.num_classes = len(set(self.label_dict.values()))

		self.seed = seed
		self.print_info = print_info
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.filter_df(slide_data, filter_dict)
		if self.multi_label:
			slide_data = self.df_prep_multi_label(slide_data, self.label_cols)
		else:
			slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def cls_ids_prep(self):
		"""
		Prepares the class ids for the dataset.
		Args:
			None
		Returns:
			None
		"""
		
		if self.multi_label:
			# store ids corresponding each class at the patient or case level
			self.patient_cls_ids = [[] for i in range(self.num_classes)]
			for i in range(self.num_classes):
				self.patient_cls_ids[i] = np.where([label[i] for label in self.patient_data['label']])[0]

			# store ids corresponding each class at the slide level
			self.slide_cls_ids = [[] for i in range(self.num_classes)]
			for i in range(self.num_classes):
				self.slide_cls_ids[i] = np.where([label[i] for label in self.slide_data['label']])[0]
		else:
			# store ids corresponding each class at the patient or case level
			self.patient_cls_ids = [[] for i in range(self.num_classes)]
			for i in range(self.num_classes):
				self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

			# store ids corresponding each class at the slide level
			self.slide_cls_ids = [[] for i in range(self.num_classes)]
			for i in range(self.num_classes):
				self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			
			if self.multi_label:
				patient_label_vecs = np.array(self.slide_data.loc[locations, 'label'].tolist())
				patient_label = (patient_label_vecs.sum(axis=0) > 0).astype(int)
				patient_labels.append(patient_label)
			else:
				label = self.slide_data['label'][locations].values
				if patient_voting == 'max':
					label = label.max() # get patient label (MIL convention)
				elif patient_voting == 'maj':
					label = stats.mode(label)[0]
				else:
					raise NotImplementedError
				patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels, dtype=object if self.multi_label else None)}

	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		"""
		Prepares the dataframe for the dataset.
		Args:
			data (pd.DataFrame): The dataframe to prepare.
			label_dict (dict): A dictionary mapping labels to integers.
			ignore (list): A list of labels to ignore.
			label_col (str): The column name of the labels.
		Returns:
			pd.DataFrame: The prepared dataframe.
		"""
		
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	@staticmethod
	def df_prep_multi_label(data, label_cols):
		data['label'] = data[label_cols].values.tolist()
		return data

	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def summarize(self):
		if self.multi_label:
			print("label columns: {}".format(self.label_cols))
		else:
			print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))

		if self.multi_label:
			# For multi-label, count occurrences of each label
			slide_labels = np.array(self.slide_data['label'].tolist())
			label_counts = slide_labels.sum(axis=0)
			print("slide-level counts:")
			for i, count in enumerate(label_counts):
				label_name = self.label_cols[i]
				print(f"  {label_name}: {count}")
		else:
			print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

	def create_splits(self, k = 3, val_num = (25, 25), test_num = (40, 40), label_frac = 1.0, custom_test_ids = None):
		settings = {
					'n_splits' : k, 
					'val_num' : val_num, 
					'test_num': test_num,
					'label_frac': label_frac,
					'seed': self.seed,
					'custom_test_ids': custom_test_ids
					}

		if self.patient_strat:
			settings.update({'cls_ids' : self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
		else:
			settings.update({'cls_ids' : self.slide_cls_ids, 'samples': len(self.slide_data)})

		self.split_gen = generate_split(**settings)

	def set_splits(self,start_from=None):
		if start_from:
			ids = nth(self.split_gen, start_from)

		else:
			ids = next(self.split_gen)

		if self.patient_strat:
			slide_ids = [[] for i in range(len(ids))] 

			for split in range(len(ids)): 
				for idx in ids[split]:
					case_id = self.patient_data['case_id'][idx]
					slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
					slide_ids[split].extend(slide_indices)

			self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]

		else:
			self.train_ids, self.val_ids, self.test_ids = ids

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, multi_label=self.multi_label)
		else:
			split = None
		
		return split

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['slide_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, multi_label=self.multi_label)
		else:
			split = None
		
		return split


	def return_splits(self, from_id=True, csv_path=None):


		if from_id:
			if len(self.train_ids) > 0:
				train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
				train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes, multi_label=self.multi_label)

			else:
				train_split = None
			
			if len(self.val_ids) > 0:
				val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
				val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes, multi_label=self.multi_label)

			else:
				val_split = None
			
			if len(self.test_ids) > 0:
				test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
				test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes, multi_label=self.multi_label)
			
			else:
				test_split = None
			
		
		else:
			assert csv_path 
			all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
			train_split = self.get_split_from_df(all_splits, 'train')
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			
		return train_split, val_split, test_split

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None

	def test_split_gen(self, return_descriptor=False):

		if return_descriptor:
			index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
			columns = ['train', 'val', 'test']
			df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index= index,
							columns= columns)

		count = len(self.train_ids)
		print('\nnumber of training samples: {}'.format(count))
		labels = self.getlabel(self.train_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'train'] = counts[u]
		
		count = len(self.val_ids)
		print('\nnumber of val samples: {}'.format(count))
		labels = self.getlabel(self.val_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'val'] = counts[u]

		count = len(self.test_ids)
		print('\nnumber of test samples: {}'.format(count))
		labels = self.getlabel(self.test_ids)
		unique, counts = np.unique(labels, return_counts=True)
		for u in range(len(unique)):
			print('number of samples in cls {}: {}'.format(unique[u], counts[u]))
			if return_descriptor:
				df.loc[index[u], 'test'] = counts[u]

		assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
		assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
		assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0

		if return_descriptor:
			return df

	def save_split(self, filename):
		train_split = self.get_list(self.train_ids)
		val_split = self.get_list(self.val_ids)
		test_split = self.get_list(self.test_ids)
		df_tr = pd.DataFrame({'train': train_split})
		df_v = pd.DataFrame({'val': val_split})
		df_t = pd.DataFrame({'test': test_split})
		df = pd.concat([df_tr, df_v, df_t], axis=1) 
		df.to_csv(filename, index = False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		label = self.getlabel(idx)
		if self.multi_label:
			label = torch.FloatTensor(label)
		slide_id = self.slide_data['slide_id'][idx]
		
		if self.use_h5:
			full_path = os.path.join(self.data_dir, 'h5_files', '{}.h5'.format(slide_id))
			if not os.path.exists(full_path):
				print(f"Warning: File not found: {full_path}")
				# Return a zero tensor as fallback or skip this sample
				# You might want to adjust this based on your needs
				return torch.zeros(1024), torch.empty(0, 2), label  # Assuming 1024 features
			
			with h5py.File(full_path, 'r') as hf:
				features = hf['features'][:]
				coords = hf['coords'][:]
			
			features = torch.from_numpy(features)
			coords = torch.from_numpy(coords)
			return features, coords, label

		else:
			full_path = os.path.join(self.data_dir, 'pt_files', slide_id+'.pt')
			if not os.path.exists(full_path):
				print(f"Warning: File not found: {full_path}")
				# Return a zero tensor as fallback or skip this sample
				# You might want to adjust this based on your needs
				return torch.zeros(1024), torch.empty(0, 2), label  # Assuming 1024 features
			features = torch.load(full_path)
			# Return dummy coordinates if not using H5, to maintain consistent output format
			return features, torch.empty(0, 2), label


class Generic_Split(Generic_MIL_Dataset):
	def __init__(self, slide_data, data_dir=None, num_classes=2, multi_label=False):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.multi_label = multi_label
		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			if self.multi_label:
				self.slide_cls_ids[i] = np.where([label[i] for label in self.slide_data['label']])[0]
			else:
				self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

	def __len__(self):
		return len(self.slide_data)
		


