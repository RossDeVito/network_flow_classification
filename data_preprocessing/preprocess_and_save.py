import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def make_stratified_splits(group_label, train_ratio, test_ratio, rand_seed=None):
	""" Makes random train, val, and test splits stratified by some 
		group_label. Returns results as a vector where 0 is train,
		1 is val, and 2 is test.
	"""
	other_inds, test_inds = next(StratifiedShuffleSplit(
		test_size=test_ratio, 
		n_splits=1, 
		random_state=rand_seed).split(group_label, group_label))
	train_inds, val_inds = next(StratifiedShuffleSplit(
		test_size=(1-test_ratio-train_ratio)/(1-test_ratio),
		n_splits=1,
		random_state=rand_seed).split(group_label[other_inds], group_label[other_inds]))

	# Make return vector of split labels
	split_labels = np.zeros(len(group_label), dtype=int)
	split_labels[other_inds[val_inds]] = 1
	split_labels[test_inds] = 2

	return split_labels
	

if __name__ == '__main__':
	# Load data
	data_path = os.path.join('..', 'data', 'darknet.csv')
	df = pd.read_csv(data_path)
	print(len(df))

	# Drop duplicate rows
	df = df.drop_duplicates()
	print(len(df))

	# Drop rows with missing values
	with pd.option_context('mode.use_inf_as_null', True):
		df = df.dropna()
	print(len(df))

	# Add binary labels
	df.loc[~df['Label'].isin(['VPN', 'Tor']), 'Label'] = 'Control'
	# binary_label - 0: Control, 1: VPN or Tor
	df['binary_label'] = df['Label'].isin(['VPN', 'Tor']).astype(int)
	df['fine_label'] = [f'{l1}_{l2}' for l1,l2 in zip(df['Label'], df['Label.1'])]
	# binary_type_label - {0: Control, 1: VPN or Tor}_{ActivityType}
	df['binary_type_label'] = [f'{l1}_{l2}' for l1,l2 in zip(df['binary_label'], df['Label.1'])]

	# Label dists
	df = df.reset_index(drop=True)
	# print(df['Label'].value_counts())
	# print(df['Label.1'].value_counts())
	print(df['binary_label'].value_counts())
	print(df['binary_type_label'].value_counts())
	# print(df['fine_label'].value_counts())


	# Make train, val, and test splits
	#  0: train, 1: val, 2: test
	df['split_1'] = make_stratified_splits(df['binary_type_label'], 0.7, 0.15, 36)
	df['split_2'] = make_stratified_splits(df['binary_type_label'], 0.7, 0.15, 147)
	df['split_3'] = make_stratified_splits(df['binary_type_label'], 0.7, 0.15, 12151997)

	# Save data
	df.to_csv(os.path.join('..', 'data', 'darknet_preprocessed.csv'), index=False)