"""
Pytorch and Pytorch Lightning modules for loading data.add()

Data modules should allow you to specify the following:
	- which label
	- which split
	- whether to use protocol feature
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


FEATURE_COLS = [
	'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
	'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
	'Fwd Packet Length Max', 'Fwd Packet Length Min',
	'Fwd Packet Length Mean', 'Fwd Packet Length Std',
	'Bwd Packet Length Max', 'Bwd Packet Length Min',
	'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
	'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
	'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
	'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
	'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
	'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
	'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
	'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
	'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
	'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
	'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
	'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
	'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
	'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
	'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
	'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
	'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
	'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
	'Idle Min',
]


class FlowDataset(Dataset):
	"""
	Dataset class for loading data.

	Args:
		label_type (str): The label type to use. 'binary' will be 0 for 
			control  and 1 for VPN/Tor. For 'both', classes are slit by 
			their 'binary' label and the activty type.
		split (str): The split to use. 'split_1' is the first split (up to _3)
		fold (int): The fold to use. If None, all folds are used. 
			Otherwise, 0: train, 1: val, 2: test
		use_protocol (bool): Whether to use the protocol feature. If True,
			the protocol feature is one-hot encoded. If False, feature
			not used.
		feature_scaler (sklearn): The optional feature scaler to use. 
			Must be already fit.
		data_file_path (str): The path to the data file.
	"""

	def __init__(self, label_type='binary', split='split_1', 
					fold=None, use_protocol=True, feature_scaler=None,
					data_file_path=os.path.join('..', 'data', 
												'darknet_preprocessed.csv')):
		"""
		Initialize the dataset.
		"""
		self.label_type = label_type
		self.split = split
		self.use_protocol = use_protocol
		self.data_file_path = data_file_path
		self.feature_scaler = feature_scaler

		# Load the data
		self.df = pd.read_csv(self.data_file_path)

		# Get the desired fold
		if fold is not None:
			self.df = self.df[self.df[self.split] == fold]

		# Get the desired label type
		if self.label_type == 'binary':
			self.labels = self.df['binary_label']
		elif self.label_type == 'both':
			raise NotImplementedError
		else:
			raise ValueError(f'Invalid label: {self.label}')

		# Subset the data to features
		self.features = self.df[FEATURE_COLS]

		# if use_protocol:
		if use_protocol:
			# make sure [0, 6, 17] one hot encoding is sufficient
			assert np.all([v in [6, 17, 0] for v in self.df["Protocol"].unique()])

			# One-hot encode the protocol
			protocol_feats = pd.get_dummies(self.df["Protocol"], prefix="protocol")
			self.features = pd.concat([self.features, protocol_feats], axis=1)

	def __len__(self):
		"""
		Return the length of the dataset.
		"""
		return len(self.labels)

	def __getitem__(self, idx):
		"""
		Return the item at the given index.
		"""
		if self.feature_scaler is not None:
			features = self.feature_scaler.transform(
				self.features.iloc[idx].values.reshape(1, -1)
			).flatten()
		else:
			features = self.features.iloc[idx].values

		return {
			'features': torch.tensor(features).float(), 
			'label': torch.tensor(self.labels.iloc[idx]).long()
		}


class FlowDataModule(pl.LightningDataModule):
	"""
	Pytorch Lightning data module.

	Args:
		label_type (str): The label type to use. 'binary' will be 0 for 
			control  and 1 for VPN/Tor. For 'both', classes are slit by 
			their 'binary' label and the activty type.
		split (str): The split to use. 'split_1' is the first split (up to _3)
		use_protocol (bool): Whether to use the protocol feature. If True,
			the protocol feature is one-hot encoded. If False, feature
			not used.
		data_file_path (str): The path to the data file.
		feature_scaler (sklearn): The optional feature scaler to use.
	"""
	def __init__(self, label_type='binary', split='split_1', 
				use_protocol=True, batch_size=64, num_workers=4,
				feature_scaler=None,
				data_file_path=os.path.join('..', 'data', 
											'darknet_preprocessed.csv')):
		super().__init__()
		self.label_type = label_type
		self.split = split
		self.use_protocol = use_protocol
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.data_file_path = data_file_path
		self.feature_scaler = feature_scaler

	def setup(self, stage=None):
		self.train_dataset = FlowDataset(
			fold=0,
			label_type=self.label_type,
			split=self.split,
			use_protocol=self.use_protocol,
			data_file_path=self.data_file_path
		)
		self.val_dataset = FlowDataset(
			fold=1,
			label_type=self.label_type,
			split=self.split,
			use_protocol=self.use_protocol,
			data_file_path=self.data_file_path
		)
		self.test_dataset = FlowDataset(
			fold=2,
			label_type=self.label_type,
			split=self.split,
			use_protocol=self.use_protocol,
			data_file_path=self.data_file_path
		)

		# Setup feature scaling
		if self.feature_scaler is not None:
			self.feature_scaler.fit(self.train_dataset.features.values)

			self.train_dataset.feature_scaler = self.feature_scaler
			self.val_dataset.feature_scaler = self.feature_scaler
			self.test_dataset.feature_scaler = self.feature_scaler

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers
		)


if __name__ == '__main__':
	__spec__ = None

	# Test the dataset
	ds = FlowDataset(label_type='binary', split='split_1', fold=0)

	# Test the data loader
	data_module = FlowDataModule(
		label_type='binary', 
		split='split_1',
		num_workers=0
	)
	data_module.setup()
	train_loader = data_module.train_dataloader()
	batch = next(iter(train_loader))

	# data_module2 = FlowDataModule(
	# 	label_type='binary', 
	# 	split='split_1',
	# 	feature_scaler=RobustScaler(),
	# 	num_workers=0
	# )
	# data_module2.setup()
	# train_loader2 = data_module2.train_dataloader()
	# batch2 = next(iter(train_loader2))

	# data_module3 = FlowDataModule(
	# 	label_type='binary', 
	# 	split='split_1',
	# 	feature_scaler=RobustScaler(unit_variance=True),
	# 	num_workers=0
	# )
	# data_module3.setup()
	# train_loader3 = data_module3.train_dataloader()
	# batch3 = next(iter(train_loader3))

	data_module4 = FlowDataModule(
		label_type='binary', 
		split='split_1',
		feature_scaler=MinMaxScaler(),
		num_workers=0
	)
	data_module4.setup()
	train_loader4 = data_module4.train_dataloader()
	batch4 = next(iter(train_loader4))
