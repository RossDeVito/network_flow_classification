import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import torch
from torch.utils.data import DataLoader
from captum import attr
from tqdm import tqdm

from nn_modules import FlowClassifier, DenseNet
from data_modules import FlowDataModule, FlowDataset


if __name__ == '__main__':
	__spec__ = None

	# Options
	model_dir = os.path.join('lightning_logs', 'version_19')
	checkpoint_fname = 'epoch=148-best_val_loss.ckpt'

	batch_size = 1024
	dl_workers = 4

	# Load model archetecture info
	with open(os.path.join(model_dir, 'best_val_res.json'), 'r') as f:
		model_info = json.load(f)

	# Load Data
	if model_info['use_scaler']:
		if model_info['type'] == 'min_max':
			scaler = MinMaxScaler()
	else:
		scaler = None

	data_module = FlowDataModule(
		label_type=model_info['label_type'],
		use_protocol=model_info['use_protocol'],
		batch_size=model_info['batch_size'],
		split=model_info['split'],
		feature_scaler=scaler,
		num_workers=0,
	)
	data_module.setup()
	full_dataset = FlowDataset(
		label_type='binary', 
		fold=None, 
		feature_scaler=data_module.feature_scaler
	)
	full_dataloader = DataLoader(
		full_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=dl_workers
	)
	
	# Load model
	net = DenseNet(
		input_size=data_module.train_dataset.features.shape[1], 
		layer_sizes=model_info['layer_sizes'],
		output_size=model_info['output_size'],
		dropout=model_info['dropout'],
	)

	weights_path = os.path.join(model_dir, 'checkpoints', checkpoint_fname)
	model = FlowClassifier.load_from_checkpoint(weights_path, network=net)
	model.eval()

	# Setup attr methods
	feat_means = full_dataset.features.mean().values
	baselines = torch.tensor(feat_means).float()

	attr_methods = {
		# 'saliency': attr.Saliency(model),
		# 'gbp': attr.GuidedBackprop(model),
		# 'ig_global': attr.IntegratedGradients(model, multiply_by_inputs=True),
		# 'ig_local': attr.IntegratedGradients(model, multiply_by_inputs=False),
		# 'deep_lift_global': attr.DeepLift(model, multiply_by_inputs=True),
		# 'deep_lift_local': attr.DeepLift(model, multiply_by_inputs=False),
		'grad_shap_global': attr.GradientShap(model, multiply_by_inputs=True),
		'grad_shap_local': attr.GradientShap(model, multiply_by_inputs=False),
	}

	# Get model predicitons
	predictions = []

	for batch in tqdm(full_dataloader, 
						total=len(full_dataloader),
						desc='Generating predictions'):
		predictions.extend(torch.sigmoid(model(batch).flatten()).tolist())

	# Generate attributions
	attr_res = {
		'model_info': model_info,
		'feat_names': full_dataset.features.columns.values,
		'original_data': full_dataset.df,
		'features': full_dataset.features,
		'labels': full_dataset.labels.values,
		'attrs': dict(),
	}

	for attr_name, attr_method in attr_methods.items():
		attrs = []

		for i, batch in tqdm(enumerate(full_dataloader), 
								total=len(full_dataloader),
								desc=f'{attr_name}'):
			
			if attr_name in ['saliency', 'gbp']:
				attr_batch = attr_method.attribute(batch['features'])
			else:
				attr_batch = attr_method.attribute(
					batch['features'], 
					baselines.reshape(1, -1)
				)

			attrs.extend(attr_batch.detach().tolist())

		attr_res['attrs'][attr_name] = np.vstack(attrs)

	# Save results
	with open(os.path.join(model_dir, 'attr_res.pkl'), 'wb') as f:
		pickle.dump(attr_res, f)
