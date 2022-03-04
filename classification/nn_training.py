"""
Train dense nueral networks to predict flow type.
"""

from datetime import datetime
import os
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import torch
import pytorch_lightning as pl

from data_modules import FlowDataModule
from nn_modules import FlowClassifier, DenseNet


TASK_TO_OUPUT_DIM = {
	'binary': 1,
}


if __name__ == '__main__':
	# Training options
	model_desc = None   # for naming saved results, if None uses timestamp
	train_opts = {
		'use_protocol': True,
		'model_type': 'dense_nn',
		'label_type': 'binary', # 'binary', 'both'
		'use_scaler': True,
		'early_stopping_patience': 10,
		'batch_size': 256,
		'learning_rate': 1e-2,
		'reduce_lr_on_plateau': False,
		'lr_patience': 7,
		'reduce_lr_factor': 0.5,
		'split': 'split_1',
		'pos_weight': [1.5], # or None
	}
	dense_opts = {
		'layer_sizes': [64, 64, 32],
		'output_size': TASK_TO_OUPUT_DIM[train_opts['label_type']],
		'dropout': 0.3,
	}
	scaler_opts = {
		'type': 'min_max',
	}

	# Load data
	if train_opts['use_scaler']:
		if scaler_opts['type'] == 'min_max':
			scaler = MinMaxScaler()
	else:
		scaler = None

	data_module = FlowDataModule(
		label_type=train_opts['label_type'],
		use_protocol=train_opts['use_protocol'],
		batch_size=train_opts['batch_size'],
		split=train_opts['split'],
		feature_scaler=scaler,
	)
	data_module.setup()

	# Create model
	net = DenseNet(
		input_size=data_module.train_dataset.features.shape[1], 
		**dense_opts
	)
	model = FlowClassifier(
		net,
		learning_rate=train_opts['learning_rate'],
		reduce_lr_on_plateau=train_opts['reduce_lr_on_plateau'],
		patience=train_opts['lr_patience'],
		reduce_lr_factor=train_opts['reduce_lr_factor'],
		pos_weight=torch.tensor(train_opts['pos_weight']) if train_opts['pos_weight'] is not None else None,
	)

	# Train model
	callbacks = [
		pl.callbacks.EarlyStopping('val_loss', verbose=True, 
			patience=train_opts['early_stopping_patience']),
		pl.callbacks.ModelCheckpoint(
			monitor="val_loss",
			filename='{epoch}-best_val_loss'
		)
	]

	trainer = pl.Trainer(
		callbacks=callbacks, 
		log_every_n_steps=1, 
		detect_anomaly=True,
		# max_epochs=3, # for debugging
	)
	trainer.fit(model, datamodule=data_module)

	# Evaluate model
	val_preds = trainer.predict(
		ckpt_path='best',
		dataloaders=data_module.val_dataloader()
	)

	test_preds = trainer.predict(
		ckpt_path='best',
		dataloaders=data_module.test_dataloader()
	)
	

	val_true = data_module.val_dataset.labels.values
	test_true = data_module.test_dataset.labels.values

	if train_opts['label_type'] == 'binary':
		val_preds = torch.sigmoid(torch.vstack(val_preds).flatten()).numpy()
		test_preds = torch.sigmoid(torch.vstack(test_preds).flatten()).numpy()

		val_preds_bin = (val_preds > 0.5).astype(int)
		test_preds_bin = (test_preds > 0.5).astype(int)

		val_class_report = classification_report(val_true, val_preds_bin, output_dict=True)
		test_class_report = classification_report(test_true, test_preds_bin, output_dict=True)

		val_cm = confusion_matrix(val_true, val_preds_bin)
		test_cm = confusion_matrix(test_true, test_preds_bin)

		val_prc = precision_recall_curve(val_true, val_preds)
		test_prc = precision_recall_curve(test_true, test_preds)

		save_dict = train_opts
		save_dict.update(dense_opts)
		save_dict.update(scaler_opts)

		save_dict['val_class_report'] = val_class_report
		save_dict['test_class_report'] = test_class_report
		save_dict['val_cm'] = val_cm.tolist()
		save_dict['test_cm'] = test_cm.tolist()
		save_dict['val_prc'] = [a.tolist() for a in val_prc]
		save_dict['test_prc'] = [a.tolist() for a in test_prc]
		save_dict['val_preds'] = val_preds.tolist()
		save_dict['test_preds'] = test_preds.tolist()
		save_dict['val_true'] = val_true.tolist()
		save_dict['test_true'] = test_true.tolist()
		save_dict['val_preds_bin'] = val_preds_bin.tolist()	# binary version of predictions
		save_dict['test_preds_bin'] = test_preds_bin.tolist()

	else:
		raise NotImplementedError

	# Save results
	with open(os.path.join(trainer.logger.log_dir, 'best_val_res.json'), 'w') as f:
		json.dump(save_dict, f, indent=4)
